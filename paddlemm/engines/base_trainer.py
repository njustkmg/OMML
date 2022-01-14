from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

import paddle
import paddle.nn as nn
from paddle.io import DataLoader

from paddlemm.models import CMML, NIC, SCAN, SGRAF, AoANet, EarlyFusion, LateFusion, VSEPP, IMRAM
from paddlemm.datasets import BasicDataset, SemiDataset, PretrainDataset, SampleDataset


DatasetMap = {
    'basic': BasicDataset,
    'semi': SemiDataset,
    'sample': SampleDataset,
    'pretrain': PretrainDataset
}

ModelMap = {
    'cmml': CMML,
    'nic': NIC,
    'scan': SCAN,
    'vsepp': VSEPP,
    'imram': IMRAM,
    'sgraf': SGRAF,
    'aoanet': AoANet,
    'earlyfusion': EarlyFusion,
    'latefusion': LateFusion
}


class BaseTrainer(metaclass=ABCMeta):

    def __init__(self, opt):

        self.model_name = opt.model_name.lower()

        self.out_root = opt.out_root
        self.logger = opt.logger

        self.num_epochs = opt.num_epochs
        self.batch_size = opt.batch_size
        self.learning_rate = opt.learning_rate
        self.task = opt.task
        self.weight_decay = opt.get('weight_decay', 0.)
        self.pretrain_epochs = opt.get('pretrain_epochs', 0)
        self.num_workers = opt.get('num_workers', 0)
        self.val_epoch = opt.get('val_epoch', 1)

        # choose metric for select best model during training
        self.select_metric = opt.get('select_metric', 'loss')

        self.dataset = DatasetMap[opt.data_mode](**opt)

        opt.vocab_size = self.dataset.vocab_size
        opt.vocab = str(self.dataset.word2idx)
        self.model = ModelMap[opt.model_name.lower()](**opt)

        self.grad_clip = opt.get('grad_clip', 0)
        if self.grad_clip:
            self.grad_clip = nn.clip.ClipGradByValue(opt.grad_clip)
        else:
            self.grad_clip = None

        self.step_size = opt.get('step_size', 0)
        self.gamma = opt.get('gamma', 0.1)
        if self.step_size:
            self.scheduler = paddle.optimizer.lr.StepDecay(learning_rate=self.learning_rate, step_size=self.step_size,
                                                           gamma=self.gamma)
            self.optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(),
                                                   learning_rate=self.scheduler,
                                                   weight_decay=self.weight_decay,
                                                   grad_clip=self.grad_clip)
        else:
            self.optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(),
                                                   learning_rate=self.learning_rate,
                                                   weight_decay=self.weight_decay,
                                                   grad_clip=self.grad_clip)

    def train(self):

        if self.pretrain_epochs > 0:
            self.pretrain()

        for epoch in range(1, self.num_epochs + 1):
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

            paddle.save(self.model.state_dict(), os.path.join(self.out_root, 'temp.pdparams'))
            if epoch % self.val_epoch == 0:
                val_res = self.evaluate()
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

    def pretrain(self):
        # for cmml pretraining

        self.model.train()
        for epoch in range(1, self.pretrain_epochs + 1):
            all_loss = []

            train_loader = DataLoader(self.dataset.train_(),
                                      batch_size=self.batch_size * 8,  # mul 8 to train total supervised data
                                      shuffle=True,
                                      num_workers=self.num_workers)
            train_tqdm = tqdm(train_loader(), ncols=80)

            for idx, batch in enumerate(train_tqdm):
                self.optimizer.clear_grad()
                loss = self.model.pretrain(batch)
                loss.backward()
                self.optimizer.step()

                all_loss.append(loss.item())
                train_tqdm.set_description("Pretrain epoch: {} | Loss: {:.3f}".format(epoch, np.mean(all_loss)))

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def test(self):
        pass
