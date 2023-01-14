import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class CMML(nn.Module):
    def __init__(self,
                 bow_dim,
                 hidden_dim,
                 num_labels,
                 **kwargs):
        super(CMML, self).__init__()

        input_dim = bow_dim
        hidden_dim = hidden_dim
        num_labels = num_labels

        # Text feature net
        self.txt_hidden = nn.Linear(input_dim, hidden_dim)
        self.txt_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        self.txt_predict = nn.Linear(hidden_dim, num_labels)

        # Image feature net
        self.resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.img_hidden = nn.Linear(512, hidden_dim)
        self.img_predict = nn.Linear(hidden_dim, num_labels)

        self.attn_mlp = nn.Linear(hidden_dim, 1)
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), 1)
        )
        self.modality_predict = nn.Linear(hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # Loss function
        self.criterion = nn.BCELoss()
        self.cita = 1.003

    def pretrain(self, batch):
        feature, label = batch['feature'], batch['label']
        supervised_txt, supervised_img, unsupervised_txt, unsupervised_img = feature
        supervised_txt = torch.cat(supervised_txt, 0)
        supervised_img = torch.cat(supervised_img, 0)
        label = torch.cat(label, 0)

        if torch.cuda.is_available():
            supervised_txt = supervised_txt.cuda()
            supervised_img = supervised_img.cuda()
            label = label.cuda()

        supervised_img_hidden = self.resnet(supervised_img)
        supervised_img_hidden = torch.reshape(supervised_img_hidden, shape=[supervised_img_hidden.shape[0], 512])
        supervised_img_hidden = self.img_hidden(supervised_img_hidden)
        supervised_img_predict = self.img_predict(supervised_img_hidden)
        supervised_img_predict = self.sigmoid(supervised_img_predict)

        supervised_txt_hidden = self.txt_hidden(supervised_txt)
        supervised_txt_predict = self.txt_predict(supervised_txt_hidden)
        supervised_txt_predict = self.sigmoid(supervised_txt_predict)

        img_loss = self.criterion(supervised_img_predict, label)
        txt_loss = self.criterion(supervised_txt_predict, label)
        loss = img_loss + txt_loss
        return loss

    def forward(self, batch):
        if self.training:
            return self.forward_train(batch)
        else:
            return self.forward_eval(batch)

    def forward_train(self, batch):
        """for training, batch contain supervised and unsupervised data"""

        feature, label = batch['feature'], batch['label']
        supervised_txt, supervised_img, unsupervised_txt, unsupervised_img = feature
        supervised_txt = torch.cat(supervised_txt, 0)
        supervised_img = torch.cat(supervised_img, 0)
        unsupervised_txt = torch.cat(unsupervised_txt, 0)
        unsupervised_img = torch.cat(unsupervised_img, 0)
        label = torch.cat(label, 0)

        if torch.cuda.is_available():
            supervised_txt = supervised_txt.cuda()
            supervised_img = supervised_img.cuda()
            unsupervised_txt = unsupervised_txt.cuda()
            unsupervised_img = unsupervised_img.cuda()
            label = label.cuda()

        # supervise training
        supervised_txt_hidden = self.txt_hidden(supervised_txt)
        supervised_txt_predict = self.txt_predict(supervised_txt_hidden)
        supervised_txt_predict = self.sigmoid(supervised_txt_predict)

        supervised_img_hidden = self.resnet(supervised_img)
        supervised_img_hidden = torch.reshape(supervised_img_hidden, shape=[supervised_img_hidden.shape[0], 512])
        supervised_img_hidden = self.img_hidden(supervised_img_hidden)
        supervised_img_predict = self.img_predict(supervised_img_hidden)
        supervised_img_predict = self.sigmoid(supervised_img_predict)

        attn_txt = self.attn_mlp(supervised_txt_hidden)
        attn_img = self.attn_mlp(supervised_img_hidden)
        attn_modality = torch.cat([attn_txt, attn_img], dim=1)
        attn_modality = self.softmax(attn_modality)

        attn_img = torch.zeros([1, len(label)])
        attn_txt = torch.zeros([1, len(label)])

        if torch.cuda.is_available():
            attn_txt = attn_txt.cuda()
            attn_img = attn_img.cuda()

        attn_img[0] = attn_modality[:, 0]
        attn_img = torch.t(attn_img)
        attn_txt[0] = attn_modality[:, 1]
        attn_txt = torch.t(attn_txt)

        supervised_hidden = attn_txt * supervised_txt_hidden + attn_img * supervised_img_hidden
        supervised_predict = self.modality_predict(supervised_hidden)
        supervised_predict = self.sigmoid(supervised_predict)

        mm_loss = self.criterion(supervised_predict, label)
        txt_loss = self.criterion(supervised_txt_predict, label)
        img_loss = self.criterion(supervised_img_predict, label)
        supervised_loss = mm_loss * 3 + txt_loss + img_loss

        # diversity measure
        similar = torch.bmm(supervised_img_predict.unsqueeze(1),
                             supervised_txt_predict.unsqueeze(2))

        similar = torch.reshape(similar, shape=[supervised_img_predict.shape[0]])
        norm_matrix_img = torch.norm(supervised_img_predict, p=2, dim=1)
        norm_matrix_text = torch.norm(supervised_txt_predict, p=2, dim=1)
        div = torch.mean(similar / (norm_matrix_img * norm_matrix_text))

        # unsupervise training
        # Robust Consistency Measure

        unsupervised_txt_hidden = self.txt_hidden(unsupervised_txt)
        unsupervised_txt_predict = self.txt_predict(unsupervised_txt_hidden)
        unsupervised_txt_predict = self.sigmoid(unsupervised_txt_predict)

        unsupervised_img_hidden = self.resnet(unsupervised_img)
        unsupervised_img_hidden = torch.reshape(unsupervised_img_hidden, shape=[unsupervised_img_hidden.shape[0], 512])
        unsupervised_img_hidden = self.img_hidden(unsupervised_img_hidden)
        unsupervised_img_predict = self.img_predict(unsupervised_img_hidden)
        unsupervised_img_predict = self.sigmoid(unsupervised_img_predict)

        unsimilar = torch.bmm(unsupervised_img_predict.unsqueeze(1),
                               unsupervised_txt_predict.unsqueeze(2))

        unsimilar = torch.reshape(unsimilar, shape=[unsupervised_img_predict.shape[0]])

        unnorm_matrix_img = torch.norm(unsupervised_img_predict, p=2, dim=1)
        unnorm_matrix_text = torch.norm(unsupervised_txt_predict, p=2, dim=1)

        dis = 2 - unsimilar / (unnorm_matrix_img * unnorm_matrix_text)

        # dis = paddle.abs(dis)
        # dis2 = paddle.sort(dis, axis=0)
        # split = int(np.sum((dis2<self.cita).numpy()))+1
        # tensor12 = dis[:split]
        # tensor22 = dis[split:]

        tensor1 = dis[torch.abs(dis) < self.cita]
        tensor2 = dis[torch.abs(dis) >= self.cita]
        tensor1loss = torch.sum(tensor1 * tensor1 / 2)
        tensor2loss = torch.sum(self.cita * (torch.abs(tensor2) - 1 / 2 * self.cita))

        unsupervised_loss = (tensor1loss + tensor2loss) / unsupervised_img.shape[0]
        total_loss = supervised_loss + 0.01 * div + unsupervised_loss

        return {
            'loss': total_loss,
        }

    def forward_eval(self, batch):
        """for evaluation, batch only contain supervised data"""

        feature, label = batch['feature'], batch['label']
        supervised_txt, supervised_img = feature

        if torch.cuda.is_available():
            supervised_txt = supervised_txt.cuda()
            supervised_img = supervised_img.cuda()
            label = label.cuda()

        supervised_txt_hidden = self.txt_hidden(supervised_txt)
        supervised_img_hidden = self.resnet(supervised_img)
        supervised_img_hidden = torch.reshape(supervised_img_hidden, shape=[supervised_img_hidden.shape[0], 512])
        supervised_img_hidden = self.img_hidden(supervised_img_hidden)

        attn_txt = self.attn_mlp(supervised_txt_hidden)
        attn_img = self.attn_mlp(supervised_img_hidden)
        attn_modality = torch.cat([attn_txt, attn_img], dim=1)
        attn_modality = self.softmax(attn_modality)

        attn_img = torch.zeros([1, len(label)])
        attn_txt = torch.zeros([1, len(label)])

        if torch.cuda.is_available():
            attn_txt = attn_txt.cuda()
            attn_img = attn_img.cuda()

        attn_img[0] = attn_modality[:, 0]
        attn_img = torch.t(attn_img)
        attn_txt[0] = attn_modality[:, 1]
        attn_txt = torch.t(attn_txt)

        supervised_hidden = attn_txt * supervised_txt_hidden + attn_img * supervised_img_hidden
        supervised_predict = self.modality_predict(supervised_hidden)
        supervised_predict = self.sigmoid(supervised_predict)

        total_loss = self.criterion(supervised_predict, label)

        return {
            'loss': total_loss,
            'logit': supervised_predict,
            'fusion_feature': supervised_hidden,
            'img_feature': supervised_img_hidden,
            'txt_feature': supervised_txt_hidden
        }