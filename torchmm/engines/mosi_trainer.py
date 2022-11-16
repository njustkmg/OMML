from tqdm import tqdm
import os
import torch
from scipy import stats
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DomfnTrainer(object):
    def __init__(self, config, model):
        self.config = config

        self.model = model

        if config.use_cuda:
            torch.device('cuda：0')
        else:
            torch.device('cpu')

        self.text_model = self.model.text_encoder
        self.vision_model = self.model.vision_encoder
        self.audio_model = self.model.audio_encoder
        self.multi_model = self.model.multi_encoder

        self.text_optim = optim.Adam(list(self.text_model.parameters()),
                                     lr=self.config.pre_lr,
                                     weight_decay=self.config.weight_decay_text)
        self.vision_optim = optim.Adam(list(self.vision_model.parameters()),
                                       lr=self.config.text_ft_lr,
                                       weight_decay=self.config.weight_decay_vision)
        self.audio_optim = optim.Adam(list(self.audio_model.parameters()),
                                      lr=self.config.pre_lr,
                                      weight_decay=self.config.weight_decay_audio)
        self.multi_optim = optim.Adam(list(self.multi_model.parameters()),
                                      lr=self.config.multi_lr,
                                      weight_decay=self.config.weight_decay_multi)
        self.text_scheduler = ReduceLROnPlateau(self.text_optim, 'min', factor=0.5,
                                                patience=self.config.patience, verbose=True)
        self.vision_scheduler = ReduceLROnPlateau(self.vision_optim, 'min', factor=0.5,
                                                  patience=self.config.patience, verbose=True)
        self.audio_scheduler = ReduceLROnPlateau(self.audio_optim, 'min', factor=0.5,
                                                 patience=self.config.patience, verbose=True)
        self.multi_scheduler = ReduceLROnPlateau(self.audio_optim, 'min', factor=0.5,
                                                 patience=self.config.patience, verbose=True)

    def train(self, train_loader):
        self.model.train()
        train_tqdm = tqdm(train_loader)
        all_out = []
        all_label = []
        all_loss = []
        if self.config.is_pretrain:
            for param_group in self.text_optim.param_groups:
                param_group['lr'] = self.config.text_ft_lr
            for param_group in self.vision_optim.param_groups:
                param_group['lr'] = self.config.vision_ft_lr
            for param_group in self.audio_optim.param_groups:
                param_group['lr'] = self.config.audio_ft_lr

        for batch in train_tqdm:
            labels = batch['labels']
            output = self.model(**batch, eta=self.config.eta)

            text_p = output['text_penalty'] / self.config.tau
            vision_p = output['vision_penalty'] / self.config.tau
            audio_p = output['audio_penalty'] / self.config.tau
            loss = output['text_loss'] + output['vision_loss'] + \
                   output['audio_loss'] + output['multi_loss']

            if text_p < self.config.gamma and vision_p < self.config.gamma and audio_p < self.config.gamma:

                self.text_optim.zero_grad()
                self.vision_optim.zero_grad()
                self.audio_optim.zero_grad()
                self.multi_optim.zero_grad()
                loss.backward()
                self.text_optim.step()
                self.vision_optim.step()
                self.audio_optim.step()
                self.multi_optim.step()
                out = output['multi_logit']
            else:
                self.text_optim.zero_grad()
                output['text_celoss'].backward(retain_graph=True)
                self.text_optim.step()
                self.vision_optim.zero_grad()
                output['vision_celoss'].backward(retain_graph=True)
                self.vision_optim.step()
                self.audio_optim.zero_grad()
                output['audio_celoss'].backward(retain_graph=True)
                self.audio_optim.step()

                text_conf, text_pred = torch.max(output['text_logit'], dim=1)
                vision_conf, vision_pred = torch.max(output['vision_logit'], dim=1)
                audio_conf, audio_pred = torch.max(output['audio_logit'], dim=1)

                if text_conf > vision_conf and text_conf > audio_conf:
                    out = output['text_logit']
                elif vision_conf > text_conf and vision_conf > audio_conf:
                    out = output['vision_logit']
                elif audio_conf > text_conf and audio_conf > vision_conf:
                    out = output['audio_logit']

            if len(out.size()) == 1:
                out = torch.unsqueeze(out, dim=0)
            all_out += out.detach().cpu().numpy().tolist()
            all_label += labels.cpu().numpy().tolist()
            all_loss.append(loss.item())
            train_tqdm.set_description('Loss: {}, text_p: {}, vision_p: {}, audio_p: {}'.format(
                np.mean(all_loss), text_p.item(), vision_p.item(), audio_p.item()))
        labels = np.array(all_label).reshape(-1)
        one_hot_targets = np.eye(self.config.num_label)[labels]
        mae = mean_absolute_error(one_hot_targets, all_out)
        return np.mean(all_loss), mae

    def pre_train(self, train_loader):
        self.model.train()
        train_tqdm = tqdm(train_loader)
        all_text_loss = []
        all_vision_loss = []
        all_audio_loss = []
        for batch in train_tqdm:
            output = self.model(**batch, eta=self.config.eta)
            loss = output['text_celoss'] + output['vision_celoss'] + output['audio_celoss']
            self.text_optim.zero_grad()
            self.vision_optim.zero_grad()
            self.audio_optim.zero_grad()
            loss.backward()
            self.text_optim.step()
            self.vision_optim.step()
            self.audio_optim.step()
            all_text_loss.append(output['text_celoss'].item())
            all_vision_loss.append(output['vision_celoss'].item())
            all_audio_loss.append(output['audio_celoss'].item())
        return np.mean(all_text_loss), np.mean(all_vision_loss), np.mean(all_audio_loss)

    def evaluate(self, model, valid_loader):
        model.eval()
        all_out = []
        all_label = []
        all_loss = []
        for batch in valid_loader:
            with torch.no_grad():
                labels = batch['labels']
                output = model(**batch, eta=self.config.eta)
                loss = output['text_loss'] + output['vision_loss'] + \
                       output['audio_loss'] + output['multi_loss']
                text_p = output['text_penalty'] / self.config.tau
                vision_p = output['vision_penalty'] / self.config.tau
                audio_p = output['audio_penalty'] / self.config.tau
                text_conf, _ = torch.max(output['text_logit'], dim=1)
                vision_conf, _ = torch.max(output['vision_logit'], dim=1)
                audio_conf, _ = torch.max(output['audio_logit'], dim=1)

                if text_p < self.config.gamma and vision_p < self.config.gamma and audio_p < self.config.gamma:
                    out = output['multi_logit'].detach().cpu().numpy()
                elif text_conf > vision_conf and text_conf > audio_conf:
                    out = output['text_logit'].detach().cpu().numpy()
                elif vision_conf > text_conf and vision_conf > audio_conf:
                    out = output['vision_logit'].detach().cpu().numpy()
                elif audio_conf > vision_conf and audio_conf > vision_conf:
                    out = output['audio_logit'].detach().cpu().numpy()
            all_loss.append(loss.item())
            all_out += out.tolist()
            all_label += labels.tolist()
        labels = np.array(all_label).reshape(-1)
        one_hot_targets = np.eye(self.config.num_label)[labels]
        mae = mean_absolute_error(one_hot_targets, all_out)
        return np.mean(all_loss), mae, all_out, all_label, one_hot_targets

    def test(self, model, test_loader, choose_threshold=0.5):

        loss, mae, all_out, all_label, one_hot_targets = self.evaluate(model, test_loader)
        corr = np.mean(np.corrcoef(all_out, one_hot_targets))
        # predict = np.array([np.array(all_out) >= choose_threshold], dtype='int')
        predict = np.argmax(all_out, axis=1)
        acc = accuracy_score(all_label, predict)
        precision = precision_score(all_label, predict, average='macro')
        recall = recall_score(all_label, predict, average='macro')
        f1 = f1_score(all_label, predict, average='macro')
        return {'mae': mae,
                'corr': corr,
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1}

    def save(self, model_path):
        torch.save(self.model, model_path)
