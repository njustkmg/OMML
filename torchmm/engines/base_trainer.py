from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch.cuda
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm

from torchmm.models import CMML, NIC, SCAN, SGRAF, AoANet, EarlyFusion, LateFusion, VSEPP, IMRAM, BFAN, TMCFusion, LMFFusion
from torchmm.datasets import BasicDataset, SemiDataset, PretrainDataset, SampleDataset, TwitterDataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DatasetMap = {
    'basic': BasicDataset,
    'semi': SemiDataset,
    'sample': SampleDataset,
    'pretrain': PretrainDataset,
    'twitter': TwitterDataset,
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
    'latefusion': LateFusion,
    'bfan': BFAN,
    'tmcfusion': TMCFusion,
    'lmffusion': LMFFusion
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

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.grad_clip = opt.get('grad_clip', 0)
        self.step_size = opt.get('step_size', 0)
        self.gamma = opt.get('gamma', 0.1)

        if self.step_size:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                             step_size=self.step_size,
                                                             gamma=self.gamma)
        else:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)

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

            train_tqdm = tqdm(train_loader, ncols=80)
            for idx, batch in enumerate(train_tqdm):
                batch['epoch'] = epoch
                loss = self.model(batch)
                loss.backward()
                if self.grad_clip:
                    clip_grad_norm(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad()

                all_loss.append(loss.cpu().item())
                train_tqdm.set_description("Epoch: {} | Loss: {:.3f}".format(epoch, loss.item()))
            train_tqdm.close()

            if self.step_size:
                self.scheduler.step()

            torch.save(self.model.state_dict(), os.path.join(self.out_root, 'temp.pkl'))
            if epoch % self.val_epoch == 0:
                val_res = self.evaluate()
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

    def pretrain(self):
        # for cmml pretraining

        self.model.train()
        for epoch in range(1, self.pretrain_epochs + 1):
            all_loss = []

            train_loader = DataLoader(self.dataset.train_(),
                                      batch_size=self.batch_size * 8,  # mul 8 to train total supervised data
                                      shuffle=True,
                                      num_workers=self.num_workers)
            train_tqdm = tqdm(train_loader, ncols=80)

            for idx, batch in enumerate(train_tqdm):
                self.optimizer.zero_grad()
                loss = self.model.pretrain(batch)
                loss.backward()
                if self.grad_clip:
                    clip_grad_norm(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                all_loss.append(loss.cpu().item())
                train_tqdm.set_description("Pretrain epoch: {} | Loss: {:.3f}".format(epoch, np.mean(all_loss)))
            train_tqdm.close()

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def test(self):
        pass
