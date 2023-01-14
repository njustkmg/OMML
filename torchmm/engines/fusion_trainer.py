from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader

from torchmm.metrics import score_fusion
from .base_trainer import BaseTrainer
from torchmm.visual import Visual
from torch.nn.utils.clip_grad import clip_grad_norm

def judge(data_mode, method, visual, choose):

    if visual.lower() == "none":
        return False
    if data_mode.lower() != "twitter":
        print("Sorry, for the time being, we can only visualize the dataset Twitter2015 or Twitter2017.")
        return False
    if method.lower() == "cmml":
        print("Sorry, for the time being, we can only visualize the method EarlyFusion, LateFusion, TMCFusion, LMFFusion.")
        return False
    if method.lower() in ["latefusion", "tmcfusion", "lmffusion"] and choose.lower() == "fusion":
        print("Sorry, the method has no fusion operation at the feature level.")
        return False

    return True


class FusionTrainer(BaseTrainer):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.best_score = 0.0
        self.best_loss = float(np.inf)
        self.step = 0
        self.if_visual_work = judge(self.opt.data_mode, self.model_name, self.opt.visual, self.opt.choose)
        if self.if_visual_work:
            self.vis = Visual(self.opt.visual, self.opt.out_root)

    def train(self):

        if self.pretrain_epochs > 0:
            self.pretrain()

        for epoch in range(1, self.num_epochs + 1):
            self.step = epoch
            all_fusion_feature = []
            all_img_feature = []
            all_txt_feature = []
            all_label_ = []
            self.model.train()

            train_loader = DataLoader(self.dataset.train_(),
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers)
            train_tqdm = tqdm(train_loader, ncols=80)

            for idx, batch in enumerate(train_tqdm):
                batch['epoch'] = epoch
                output = self.model(batch)
                loss = output['loss']
                loss.backward()
                if self.if_visual_work:
                    img_feature = output['img_feature']
                    txt_feature = output['txt_feature']
                    all_img_feature += img_feature.cpu().tolist()
                    all_txt_feature += txt_feature.cpu().tolist()
                    all_label_ += batch['label_'].tolist()
                    if self.opt.choose == "fusion":
                        fusion_feature = output['fusion_feature']
                        all_fusion_feature += fusion_feature.cpu().tolist()

                if self.grad_clip:
                    clip_grad_norm(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad()

                train_tqdm.set_description("Epoch: {} | Loss: {:.3f}".format(epoch, loss.item()))
            train_tqdm.close()

            if self.if_visual_work:
                all_img_feature = np.array(all_img_feature)
                all_txt_feature = np.array(all_txt_feature)
                all_label_ = np.array(all_label_)
                if self.opt.choose == "fusion":
                    all_fusion_feature = np.array(all_fusion_feature)
                    self.vis.plot([all_fusion_feature], all_label_, "train_fusion", epoch, False)
                elif self.opt.choose == "image":
                    self.vis.plot([all_img_feature], all_label_, "train_image", epoch, False)
                elif self.opt.choose == "text":
                    self.vis.plot([all_txt_feature], all_label_, "train_text", epoch, False)
                else:
                    self.vis.plot([all_img_feature, all_txt_feature], all_label_, "train_image&text", epoch, False)

            if self.step_size:
                self.scheduler.step()

            torch.save(self.model.state_dict(), os.path.join(self.out_root, 'temp.pkl'))
            if epoch % self.val_epoch == 0:
                self.step = epoch
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

    def evaluate(self):
        valid_loss = []
        all_fusion_feature = []
        all_img_feature = []
        all_txt_feature = []
        all_prediction = []
        all_label = []
        all_label_ = []

        self.model.eval()
        valid_loader = DataLoader(self.dataset.valid_(),
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        valid_tqdm = tqdm(valid_loader, ncols=80)
        for idx, batch in enumerate(valid_tqdm):
            with torch.no_grad():
                batch['epoch'] = self.step
                output = self.model(batch)
                loss = output['loss']
                logit = output['logit']
                valid_loss.append(loss.item())
                valid_loss.append(loss.cpu().item())
                all_prediction += logit.cpu().tolist()
                all_label += batch['label'].cpu().tolist()
                if self.if_visual_work:
                    img_feature = output['img_feature']
                    txt_feature = output['txt_feature']
                    all_img_feature += img_feature.cpu().tolist()
                    all_txt_feature += txt_feature.cpu().tolist()
                    all_label_ += batch['label_'].tolist()
                    if self.opt.choose == "fusion":
                        fusion_feature = output['fusion_feature']
                        all_fusion_feature += fusion_feature.cpu().tolist()

        valid_tqdm.close()

        all_prediction = np.array(all_prediction)
        all_label = np.array(all_label)

        if self.if_visual_work:
            all_img_feature = np.array(all_img_feature)
            all_txt_feature = np.array(all_txt_feature)
            all_label_ = np.array(all_label_)
            if self.opt.choose == "fusion":
                all_fusion_feature = np.array(all_fusion_feature)
                self.vis.plot([all_fusion_feature], all_label_, "valid_fusion", 0, False)
            elif self.opt.choose == "image":
                self.vis.plot([all_img_feature], all_label_, "valid_image", 0, False)
            elif self.opt.choose == "text":
                self.vis.plot([all_txt_feature], all_label_, "valid_text", 0, False)
            else:
                self.vis.plot([all_img_feature, all_txt_feature], all_label_, "valid_image&text", 0, False)

        val_res = {'loss': float(np.mean(valid_loss))}

        if self.select_metric != 'loss':
            score = score_fusion(all_label, all_prediction)
            val_res.update(score)

        return val_res

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.out_root, 'best_model.pkl')))
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()
        all_prediction = []
        all_fusion_feature = []
        all_img_feature = []
        all_txt_feature = []
        all_label = []
        all_label_ = []

        test_loader = DataLoader(self.dataset.test_(),
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)
        test_tqdm = tqdm(test_loader, ncols=80)
        for idx, batch in enumerate(test_tqdm):
            with torch.no_grad():
                batch['epoch'] = self.opt.num_epochs
                output = self.model(batch)
                logit = output['logit']
                all_prediction += logit.cpu().tolist()
                all_label += batch['label'].cpu().tolist()
                if self.if_visual_work:
                    img_feature = output['img_feature']
                    txt_feature = output['txt_feature']
                    all_img_feature += img_feature.cpu().tolist()
                    all_txt_feature += txt_feature.cpu().tolist()
                    all_label_ += batch['label_'].tolist()
                    if self.opt.choose == "fusion":
                        fusion_feature = output['fusion_feature']
                        all_fusion_feature += fusion_feature.cpu().tolist()
        test_tqdm.close()

        all_prediction = np.array(all_prediction)
        all_label = np.array(all_label)
        result = score_fusion(torch.FloatTensor(all_label), all_prediction)

        for k, v in result.items():
            self.logger.info(f"{k}: {str(v)}")

        if self.if_visual_work:
            all_img_feature = np.array(all_img_feature)
            all_txt_feature = np.array(all_txt_feature)
            all_label_ = np.array(all_label_)
            if self.opt.choose == "fusion":
                all_fusion_feature = np.array(all_fusion_feature)
                self.vis.plot([all_fusion_feature], all_label_, "test_fusion", 0, True)
            elif self.opt.choose == "image":
                self.vis.plot([all_img_feature], all_label_, "test_image", 0, True)
            elif self.opt.choose == "text":
                self.vis.plot([all_txt_feature], all_label_, "test_text", 0, True)
            else:
                self.vis.plot([all_img_feature, all_txt_feature], all_label_, "test_image&text", 0, True)
