from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import os

import paddle
from paddle.io import DataLoader

from paddlemm.metrics import score_fusion
from .base_trainer import BaseTrainer


class FusionTrainer(BaseTrainer):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.best_score = 0.0
        self.best_loss = float(np.inf)

    def evaluate(self):
        valid_loss = []
        all_prediction = []
        all_label = []

        self.model.eval()
        valid_loader = DataLoader(self.dataset.valid_(),
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        valid_tqdm = tqdm(valid_loader(), ncols=80)
        for idx, batch in enumerate(valid_tqdm):
            with paddle.no_grad():
                batch['epoch'] = self.opt.num_epochs
                loss, logit = self.model(batch)
                valid_loss.append(loss.item())
                all_prediction += logit.tolist()
                all_label += batch['label'].tolist()
        valid_tqdm.close()

        all_prediction = np.array(all_prediction)
        all_label = np.array(all_label)

        val_res = {'loss': float(np.mean(valid_loss))}

        if self.select_metric != 'loss':
            score = score_fusion(all_label, all_prediction)
            val_res.update(score)

        return val_res

    def test(self):

        checkpoint = paddle.load(os.path.join(self.out_root, 'best_model.pdparams'))
        self.model.set_state_dict(checkpoint)
        self.model.eval()
        all_prediction = []
        all_label = []

        test_loader = DataLoader(self.dataset.test_(),
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)
        test_tqdm = tqdm(test_loader(), ncols=80)
        for idx, batch in enumerate(test_tqdm):
            with paddle.no_grad():
                batch['epoch'] = self.opt.num_epochs
                _, logit = self.model(batch)

                all_prediction += logit.tolist()
                all_label += batch['label'].tolist()
        test_tqdm.close()

        all_prediction = np.array(all_prediction)
        all_label = np.array(all_label)

        result = score_fusion(all_label, all_prediction)

        for k, v in result.items():
            self.logger.info(f"{k}: {str(v)}")
