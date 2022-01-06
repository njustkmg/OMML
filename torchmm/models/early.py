import torch
import torch.nn as nn
from .backbone import Vgg, Resnet, Lstm, Gru


ModelMap = {
    'vgg': Vgg,
    'resnet': Resnet,
    'lstm': Lstm,
    'gru': Gru
}


class EarlyFusion(nn.Module):

    def __init__(self,
                 image_model,
                 text_model,
                 word_dim,
                 vocab_size,
                 hidden_dim,
                 num_labels,
                 model_name,
                 option,
                 finetune,
                 **kwargs):
        super(EarlyFusion, self).__init__()

        self.option = option

        self.image_model = ModelMap[image_model.lower()](hidden_dim, num_labels, model_name, finetune)
        self.text_model = ModelMap[text_model.lower()](word_dim, hidden_dim, num_labels, vocab_size, model_name)

        self.linear1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )
        self.linear2 = nn.Linear(hidden_dim, num_labels)

        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):

        img_hidden = self.image_model(batch)
        txt_hidden = self.text_model(batch)

        if self.option == "concat":
            hidden = torch.cat([img_hidden, txt_hidden], 1)
            predict = self.linear1(hidden)
        else:
            hidden = torch.add(img_hidden, txt_hidden)
            predict = self.linear2(hidden)

        predict = self.sigmoid(predict)

        loss = self.criterion(predict, batch['label'])

        if self.training:
            return loss
        else:
            return loss, predict