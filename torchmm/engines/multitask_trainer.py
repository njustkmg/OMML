from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from torchmm.models import VILBERTPretrain, VILBERTFinetune, BertConfig
from torchmm.datasets import PretrainDataset, SampleDataset


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
            self.criterion = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.grad_clip = opt.get('grad_clip', 0)
        self.warmup_proportion = opt.get('warmup_proportion', 0.1)

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.98),
        )

        num_train_optimization_steps = int(len(self.dataset.train_()) / self.batch_size) * self.num_epochs

        self.scheduler = WarmupLinearSchedule(
            self.optimizer,
            warmup_steps=self.warmup_proportion * num_train_optimization_steps,
            t_total=num_train_optimization_steps,
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
        data_tqdm = tqdm(data_loader)
        for idx, batch in enumerate(data_tqdm):
            target = torch.zeros(self.batch_size, dtype=torch.int64)
            image_feat = batch['image_feat'].reshape([-1, batch['image_feat'].shape[2], batch['image_feat'].shape[3]])
            image_loc = batch['image_loc'].reshape([-1, batch['image_loc'].shape[2], batch['image_loc'].shape[3]])
            image_mask = batch['image_mask'].reshape([-1, batch['image_mask'].shape[2]])
            text_token = batch['text_token'].reshape([-1, batch['text_token'].shape[2]])
            text_mask = batch['text_mask'].reshape([-1, batch['text_mask'].shape[2]])
            text_segment = torch.zeros_like(text_mask, dtype=torch.int64)
            co_attention_mask = torch.zeros([text_token.shape[0], image_mask.shape[1], text_token.shape[1]])

            if torch.cuda.is_available():
                target = target.cuda()
                image_feat = image_feat.cuda()
                image_loc = image_loc.cuda()
                image_mask = image_mask.cuda()
                text_token = text_token.cuda()
                text_mask = text_mask.cuda()
                text_segment = text_segment.cuda()
                co_attention_mask = co_attention_mask.cuda()

            with torch.no_grad():
                vil_prediction, vil_logit, vil_binary_prediction = \
                    self.model(text_token, image_feat, image_loc, text_segment, text_mask, image_mask, co_attention_mask)
                vil_logit = torch.reshape(vil_logit, shape=[self.batch_size, 3])

                loss = self.criterion(vil_logit, target)
                all_loss.append(loss.item())
                preds = torch.argmax(vil_logit, 1)
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
            train_tqdm = tqdm(train_loader)

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

                if torch.cuda.is_available():
                    text_token = text_token.cuda()
                    text_mask = text_mask.cuda()
                    text_segment = text_segment.cuda()
                    text_label = text_label.cuda()
                    is_next = is_next.cuda()
                    image_feat = image_feat.cuda()
                    image_loc = image_loc.cuda()
                    image_label = image_label.cuda()
                    image_target = image_target.cuda()
                    image_mask = image_mask.cuda()

                image_label = image_label * (is_next == 0).long().unsqueeze(1)
                image_label[image_label == 0] = -1
                text_label = text_label * (is_next == 0).long().unsqueeze(1)
                text_label[text_label == 0] = -1

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
                self.optimizer.zero_grad()

                train_tqdm.set_description("Epoch: {} | Loss: {:.3f}".format(epoch, np.mean(all_loss)))
            train_tqdm.close()

        torch.save(self.model.state_dict(), os.path.join(self.out_root, 'best_model.pkl'))

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
            train_tqdm = tqdm(train_loader)

            for idx, batch in enumerate(train_tqdm):
                target = torch.zeros(self.batch_size, dtype=torch.int64)
                image_feat = batch['image_feat'].reshape([-1, batch['image_feat'].shape[2], batch['image_feat'].shape[3]])
                image_loc = batch['image_loc'].reshape([-1, batch['image_loc'].shape[2], batch['image_loc'].shape[3]])
                image_mask = batch['image_mask'].reshape([-1, batch['image_mask'].shape[2]])
                text_token = batch['text_token'].reshape([-1, batch['text_token'].shape[2]])
                text_mask = batch['text_mask'].reshape([-1, batch['text_mask'].shape[2]])
                text_segment = torch.zeros_like(text_mask, dtype=torch.int64)
                co_attention_mask = torch.zeros([text_token.shape[0], image_mask.shape[1], text_token.shape[1]])

                if torch.cuda.is_available():
                    target = target.cuda()
                    image_feat = image_feat.cuda()
                    image_loc = image_loc.cuda()
                    image_mask = image_mask.cuda()
                    text_token = text_token.cuda()
                    text_mask = text_mask.cuda()
                    text_segment = text_segment.cuda()
                    co_attention_mask = co_attention_mask.cuda()

                vil_prediction, vil_logit, vil_binary_prediction = \
                    self.model(text_token, image_feat, image_loc, text_segment, text_mask, image_mask, co_attention_mask)
                vil_logit = torch.reshape(vil_logit, shape=[self.batch_size, 3])

                loss = self.criterion(vil_logit, target)
                loss.backward()
                all_loss.append(loss.item())

                preds = torch.argmax(vil_logit, 1)
                all_pred += preds.cpu().detach().numpy().tolist()
                all_label += target.cpu().squeeze().tolist()

                train_tqdm.set_description("Epoch: {} | Loss: {:.3f} | Acc: {:.3f}"
                                           .format(epoch, float(np.mean(all_loss)),
                                                   float((np.array(all_pred) == np.array(all_label)).sum()) / float(
                                                       len(all_label))))

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            train_tqdm.close()

            if epoch % self.val_epoch == 0:
                data_loader = DataLoader(self.dataset.valid_(), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                val_res = self.evaluate(data_loader)

                if self.select_metric == 'loss':
                    if val_res['loss'] < self.best_loss:
                        self.best_loss = val_res['loss']
                        torch.save(self.model.state_dict(), os.path.join(self.out_root, 'best_model.pkl'))
                    self.logger.info("Epoch: {}, valid loss: {:.3f}, Best: {:.3f}".format(epoch, val_res['loss'], self.best_loss))
                else:
                    if val_res[self.select_metric] > self.best_score:
                        self.best_score = val_res[self.select_metric]
                        torch.save(self.model.state_dict(), os.path.join(self.out_root, 'best_model.pkl'))
                    self.logger.info("Epoch: {}, valid score: {:.3f}, Best: {:.3f}".format(epoch, val_res[self.select_metric],
                                                                                self.best_score))

        self.model.load_state_dict(torch.load(os.path.join(self.out_root, 'best_model.pkl')))
        self.model.eval()
        data_loader = DataLoader(self.dataset.test_(), batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers)
        test_result = self.evaluate(data_loader)
        for k, v in test_result.items():
            self.logger.info(f"{k}: {str(v)}")

