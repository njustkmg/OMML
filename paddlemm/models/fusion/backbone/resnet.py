import paddle
import paddle.nn as nn
from paddle.vision import models


class Resnet(nn.Layer):

    def __init__(self, hidden_dim, num_labels, model_name, finetune=False):
        super(Resnet, self).__init__()

        hidden_dim = hidden_dim
        num_labels = num_labels

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

        img_hidden = self.resnet(batch['image_feat'])
        img_hidden = paddle.squeeze(img_hidden, (2, 3))
        img_hidden = self.hidden(img_hidden)
        img_predict = self.predict(img_hidden)

        if self.model_name == "earlyfusion" or self.model_name == "lmffusion":
            return img_hidden
        elif self.model_name == "latefusion" or self.model_name == "tmcfusion":
            return img_predict
        else:
            img_predict = self.sigmoid(img_predict)
            loss = self.criterion(img_predict, batch['label'])
            if self.training:
                return loss
            else:
                return loss, img_predict