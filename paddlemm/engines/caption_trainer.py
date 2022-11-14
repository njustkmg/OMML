from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import os

import paddle
from paddle.io import DataLoader

from paddlemm.metrics import score_caption
from .base_trainer import BaseTrainer


class CaptionTrainer(BaseTrainer):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.best_score = 0.0
        self.best_loss = float(np.inf)

    def train(self):
        if self.opt.model_name.lower() == 'aoanet':
            self.train_xe()
        else:
            BaseTrainer.train(self)

    def train_xe(self):

        for epoch in range(1, self.num_epochs + 1):

            if epoch > 1:
                # Assign the scheduled sampling prob
                frac = epoch // 5
                ss_prob = min(0.05 * frac, 0.5)
                self.model.ss_prob = ss_prob

            all_loss = []
            self.model.train()

            train_loader = DataLoader(self.dataset.train_(),
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers)

            train_tqdm = tqdm(train_loader(), ncols=80)
            for idx, batch in enumerate(train_tqdm):
                loss = self.model(batch)
                loss.backward()

                self.optimizer.step()
                self.optimizer.clear_grad()

                all_loss.append(loss.item())
                train_tqdm.set_description("Epoch: {} | Loss: {:.3f}".format(epoch, loss.item()))
            train_tqdm.close()

            if self.step_size:
                self.scheduler.step()

            paddle.save(self.model.state_dict(), os.path.join(self.out_root, 'temp_ce.pdparams'))
            if epoch % self.val_epoch == 0:
                val_res = self.evaluate()
                if val_res[self.select_metric] > self.best_score:
                    self.best_score = val_res[self.select_metric]
                    paddle.save(self.model.state_dict(), os.path.join(self.out_root, 'best_ce_model.pdparams'))
                self.logger.info("Epoch: {}, valid score: {:.3f}, Best: {:.3f}".format(epoch, val_res[self.select_metric], self.best_score))


    def evaluate(self):

        idx2word = self.dataset.idx2word
        word2idx = self.dataset.word2idx

        origin_text = []
        pred_text = []

        self.model.eval()
        valid_loader = DataLoader(self.dataset.valid_(),
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)
        valid_tqdm = tqdm(valid_loader(), ncols=80)
        for idx, batch in enumerate(valid_tqdm):
            with paddle.no_grad():
                _, logit = self.model(batch)

                gts_caption = []
                all_caption = batch['all_text']
                for i in range(len(all_caption[0])):
                    gts_caption.append([cap[i] for cap in all_caption])
                origin_text += gts_caption

                word_idx = logit.tolist()
                for words in word_idx:
                    sent = [idx2word[idx] for idx in words
                            if idx != word2idx['<start>'] and idx != word2idx['<pad>'] and idx != word2idx['<end>']]
                    pred_text.append(' '.join(sent))
        valid_tqdm.close()

        if self.opt.model_name.lower() == 'aoanet':
            pred_text = [pred_text[_] for _ in range(0, len(pred_text), 5)]

        gts = {}
        res = {}
        for i in range(len(pred_text)):
            gts[i] = origin_text[i]
            res[i] = [pred_text[i]]

        val_res = score_caption(gts, res)

        return val_res

    def test(self):

        checkpoint = paddle.load(os.path.join(self.out_root, 'best_model.pdparams'))
        self.model.set_state_dict(checkpoint)
        self.model.eval()

        idx2word = self.dataset.idx2word
        word2idx = self.dataset.word2idx

        origin_text = []
        pred_text = []

        self.model.eval()
        test_loader = DataLoader(self.dataset.test_(),
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)
        test_tqdm = tqdm(test_loader(), ncols=80)
        for idx, batch in enumerate(test_tqdm):
            with paddle.no_grad():
                _, logit = self.model(batch)

                gts_caption = []
                all_caption = batch['all_text']
                for i in range(len(all_caption[0])):
                    gts_caption.append([cap[i] for cap in all_caption])
                origin_text += gts_caption

                word_idx = logit.tolist()
                for words in word_idx:
                    sent = [idx2word[idx] for idx in words
                            if idx != word2idx['<start>'] and idx != word2idx['<pad>'] and idx != word2idx['<end>']]
                    pred_text.append(' '.join(sent))
        test_tqdm.close()

        if self.opt.model_name.lower() == 'aoanet':
            pred_text = [pred_text[_] for _ in range(0, len(pred_text), 5)]

        gts = {}
        res = {}
        for i in range(len(pred_text)):
            gts[i] = origin_text[i]
            res[i] = [pred_text[i]]

        result = score_caption(gts, res)
        for k, v in result.items():
            self.logger.info(f"{k}: {str(v)}")