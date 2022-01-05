from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import os

import paddle
from paddle.io import DataLoader

from paddlemm.models import VILBERTPretrain, VILBERTFinetune, BertConfig
from paddlemm.datasets import PretrainDataset, SampleDataset
from .tools.constdecay_warmup import ConstDecayWithWarmup


DatasetMap = {
    'sample': SampleDataset,
    'pretrain': PretrainDataset
}


class MultitaskTrainer(object):

    def __init__(self, opt):

        self.model_name = opt.model_name.lower()
        self.logger = opt.logger

        if not os.path.exists(opt.out_root):
            os.mkdir(opt.out_root)
        self.out_root = opt.out_root

        self.num_epochs = opt.num_epochs
        self.batch_size = opt.batch_size
        self.learning_rate = opt.learning_rate
        self.sub_task = opt.sub_task
        self.weight_decay = opt.get('weight_decay', 0.)
        self.num_workers = opt.get('num_workers', 0)
        self.val_epoch = opt.get('val_epoch', 1)

        # choose metric for select best model during training
        self.select_metric = opt.get('select_metric', 'loss')

        self.dataset = DatasetMap[opt.data_mode](**opt)
        self.bert_config = BertConfig.from_json_file(opt.bert_config)

        if self.sub_task == 'pretrain':
            self.model = VILBERTPretrain(self.bert_config)
        else:
            self.model = VILBERTFinetune(self.bert_config, num_labels=opt.num_labels)
            self.criterion = paddle.nn.CrossEntropyLoss()

        self.grad_clip = opt.get('grad_clip', 0)
        self.warmup_proportion = opt.get('warmup_proportion', 0.1)

        # load optimizer
        if self.sub_task == 'pretrain':
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

            num_train_optimization_steps = int(len(self.dataset.train_()) / self.batch_size) * self.num_epochs

            self.scheduler = ConstDecayWithWarmup(learning_rate=self.learning_rate,
                                                  warmup=self.warmup_proportion,
                                                  total_steps=num_train_optimization_steps)
            self.optimizer = paddle.optimizer.AdamW(parameters=self.model.parameters(),
                                                    learning_rate=self.scheduler,
                                                    epsilon=1e-8,
                                                    beta1=0.9,
                                                    beta2=0.98,
                                                    weight_decay=self.weight_decay,
                                                    apply_decay_param_fun=lambda x:
                                                    x in [p.name for n, p in self.model.named_parameters()
                                                          if not any(nd in n for nd in no_decay)],
                                                    grad_clip=paddle.nn.clip.ClipGradByValue(self.grad_clip))

        else:
            no_finetune = ['vil_prediction', 'vil_logit', 'vision_logit', 'linguistic_logit']
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

            # reduce lr on this epoch
            lr_reduce_list = [12, 16]
            num_train_optimization_steps = int(len(self.dataset.train_()) / self.batch_size) * self.num_epochs
            decay_steps = [(decay_epoch / float(self.num_epochs) * num_train_optimization_steps) for decay_epoch in lr_reduce_list]

            # set up optimizer
            # train additional classifiers of downstream tasks from sratch
            self.from_scratch_scheduler = ConstDecayWithWarmup(1e-4, self.warmup_proportion, decay_steps,
                                                               num_train_optimization_steps)
            self.from_scratch_optimizer = paddle.optimizer.AdamW(
                learning_rate=self.from_scratch_scheduler,
                parameters=[p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_finetune)],
                weight_decay=self.weight_decay,
                apply_decay_param_fun=lambda x: x in [
                    p.name for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_finetune)
                       and any(nd in n for nd in no_decay)
                ],
                grad_clip=paddle.fluid.clip.ClipGradByValue(self.grad_clip)
            )

            # fintune pretrained ViLBERT with slow learning rate
            self.finetune_scheduler = ConstDecayWithWarmup(self.learning_rate, self.warmup_proportion,
                                                           decay_steps, num_train_optimization_steps)
            self.finetune_optimizer = paddle.optimizer.AdamW(
                learning_rate=self.finetune_scheduler,
                parameters=[p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_finetune)],
                weight_decay=self.weight_decay,
                apply_decay_param_fun=lambda x: x in [
                    p.name for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_finetune)
                       and any(nd in n for nd in no_decay)
                ],
                grad_clip=paddle.fluid.clip.ClipGradByValue(self.grad_clip)
            )

        self.best_score = 0.0
        self.best_loss = float(np.inf)

    def train(self):
        if self.sub_task == 'pretrain':
            self.pretrain()
        else:
            self.finetune()

    def test(self):
        pass

    def evaluate(self, data_loader):
        all_loss = []
        all_label = []
        all_pred = []

        self.model.eval()
        data_tqdm = tqdm(data_loader())
        for idx, batch in enumerate(data_tqdm):
            target = paddle.zeros([self.batch_size], dtype='int64')
            image_feat = batch['image_feat'].reshape([-1, batch['image_feat'].shape[2], batch['image_feat'].shape[3]])
            image_loc = batch['image_loc'].reshape([-1, batch['image_loc'].shape[2], batch['image_loc'].shape[3]])
            image_mask = batch['image_mask'].reshape([-1, batch['image_mask'].shape[2]])
            text_token = batch['text_token'].reshape([-1, batch['text_token'].shape[2]])
            text_mask = batch['text_mask'].reshape([-1, batch['text_mask'].shape[2]])
            text_segment = paddle.zeros_like(text_mask, dtype='int64')
            co_attention_mask = paddle.zeros([text_token.shape[0], image_mask.shape[1], text_token.shape[1]])

            with paddle.no_grad():
                vil_prediction, vil_logit, vil_binary_prediction = \
                    self.model(text_token, image_feat, image_loc, text_segment, text_mask, image_mask, co_attention_mask)
                vil_logit = paddle.reshape(vil_logit, shape=[self.batch_size, 3])

                loss = self.criterion(vil_logit, target)
                all_loss.append(loss.item())
                preds = paddle.argmax(vil_logit, 1)
                all_pred += preds.cpu().detach().numpy().tolist()
                all_label += target.cpu().squeeze().tolist()
        data_tqdm.close()

        valid_acc = float((np.array(all_pred) == np.array(all_label)).sum()) / float(len(all_label))
        return {'acc': valid_acc, 'loss': np.mean(all_loss)}

    def pretrain(self):

        for epoch in range(1, self.num_epochs + 1):
            all_loss = []

            self.model.train()
            train_loader = DataLoader(self.dataset.train_(), batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)
            train_tqdm = tqdm(train_loader())

            for idx, batch in enumerate(train_tqdm):
                text_token = batch['text_token']
                text_mask = batch['text_mask']
                text_segment = batch['text_segment']
                text_label = batch['text_label']
                is_next = batch['is_next']
                image_feat = batch['image_feat']
                image_loc = batch['image_loc']
                image_target = batch['image_target']
                image_label = batch['image_label']
                image_mask = batch['image_mask']

                # image_label = image_label * (is_next == 0).unsqueeze(1)
                # image_label[image_label == 0] = -1
                # text_label = text_label * (is_next == 0).unsqueeze(1)
                # text_label[text_label == 0] = -1

                masked_loss_t, masked_loss_v, next_sentence_loss = self.model(
                    text_token,
                    image_feat,
                    image_loc,
                    text_segment,
                    text_mask,
                    image_mask,
                    text_label,
                    image_label,
                    image_target,
                    is_next,
                )
                loss = masked_loss_t + next_sentence_loss
                loss.backward()

                all_loss.append(loss.item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.clear_grad()

                train_tqdm.set_description("Epoch: {} | Loss: {:.3f}".format(epoch, np.mean(all_loss)))
            train_tqdm.close()

        paddle.save(self.model.state_dict(), os.path.join(self.out_root, 'best_model.pdparams'))

    def finetune(self):

        for epoch in range(1, self.num_epochs + 1):
            all_loss = []
            all_label = []
            all_pred = []

            self.model.train()

            train_loader = DataLoader(self.dataset.train_(),
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers)
            train_tqdm = tqdm(train_loader())

            for idx, batch in enumerate(train_tqdm):
                target = paddle.zeros([self.batch_size], dtype='int64')
                image_feat = batch['image_feat'].reshape([-1, batch['image_feat'].shape[2], batch['image_feat'].shape[3]])
                image_loc = batch['image_loc'].reshape([-1, batch['image_loc'].shape[2], batch['image_loc'].shape[3]])
                image_mask = batch['image_mask'].reshape([-1, batch['image_mask'].shape[2]])
                text_token = batch['text_token'].reshape([-1, batch['text_token'].shape[2]])
                text_mask = batch['text_mask'].reshape([-1, batch['text_mask'].shape[2]])
                text_segment = paddle.zeros_like(text_mask, dtype='int64')
                co_attention_mask = paddle.zeros([text_token.shape[0], image_mask.shape[1], text_token.shape[1]])

                vil_prediction, vil_logit, vil_binary_prediction = \
                    self.model(text_token, image_feat, image_loc, text_segment, text_mask, image_mask, co_attention_mask)
                vil_logit = paddle.reshape(vil_logit, shape=[self.batch_size, 3])

                loss = self.criterion(vil_logit, target)
                loss.backward()
                all_loss.append(loss.item())

                preds = paddle.argmax(vil_logit, 1)
                all_pred += preds.cpu().detach().numpy().tolist()
                all_label += target.cpu().squeeze().tolist()

                train_tqdm.set_description("Epoch: {} | Loss: {:.3f} | Acc: {:.3f}"
                                           .format(epoch, float(np.mean(all_loss)),
                                                   float((np.array(all_pred) == np.array(all_label)).sum()) / float(
                                                       len(all_label))))

                self.finetune_scheduler.step()
                self.from_scratch_scheduler.step()
                self.finetune_optimizer.step()
                self.from_scratch_optimizer.step()

                self.finetune_optimizer.clear_grad()
                self.from_scratch_optimizer.clear_grad()
            train_tqdm.close()

            if epoch % self.val_epoch == 0:
                data_loader = DataLoader(self.dataset.valid_(), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                val_res = self.evaluate(data_loader)

                if self.select_metric == 'loss':
                    if val_res['loss'] < self.best_loss:
                        self.best_loss = val_res['loss']
                        paddle.save(self.model.state_dict(), os.path.join(self.out_root, 'best_model.pdparams'))
                    self.logger.info("Epoch: {}, valid loss: {:.3f}, Best: {:.3f}".format(epoch, val_res['loss'], self.best_loss))
                else:
                    if val_res[self.select_metric] > self.best_score:
                        self.best_score = val_res[self.select_metric]
                        paddle.save(self.model.state_dict(), os.path.join(self.out_root, 'best_model.pdparams'))
                    self.logger.info("Epoch: {}, valid score: {:.3f}, Best: {:.3f}".format(epoch, val_res[self.select_metric],
                                                                                self.best_score))

        self.model.load_state_dict(paddle.load(os.path.join(self.out_root, 'best_model.pdparams')))
        self.model.eval()
        data_loader = DataLoader(self.dataset.test_(), batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers)
        test_result = self.evaluate(data_loader)
        for k, v in test_result.items():
            self.logger.info(f"{k}: {str(v)}")

