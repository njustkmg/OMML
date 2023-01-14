import torch
import torch.nn as nn
from torchvision import models


class Resnet(nn.Module):

    def __init__(self, hidden_dim, num_labels, model_name, finetune=False):
        super(Resnet, self).__init__()

        self.model_name = model_name.lower()
        self.resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.hidden = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
        self.predict = nn.Linear(hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        if not finetune:
            for p in self.resnet.parameters():
                p.require_grads = False

    def forward(self, batch):

        if torch.cuda.is_available():
            batch['image_feat'] = batch['image_feat'].cuda()
            batch['label'] = batch['label'].cuda()

        img_resnet = self.resnet(batch['image_feat'])
        img_resnet = torch.squeeze(img_resnet, 2)
        img_resnet = torch.squeeze(img_resnet, 2)
        img_hidden = self.hidden(img_resnet)
        img_predict = self.predict(img_hidden)

        if self.model_name == "earlyfusion" or self.model_name == "lmffusion":
            return img_hidden
        elif self.model_name == "latefusion" or self.model_name == "tmcfusion":
            return img_hidden, img_predict
        else:
            img_predict = self.sigmoid(img_predict)
            loss = self.criterion(img_predict, batch['label'])
            if self.training:
                return loss
            else:
                return loss, img_predict