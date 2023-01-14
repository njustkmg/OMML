import torch
import torch.nn as nn
from torchmm.models.fusion.backbone import Vgg, Resnet, Lstm, Gru


ModelMap = {
    'vgg': Vgg,
    'resnet': Resnet,
    'lstm': Lstm,
    'gru': Gru
}


class LateFusion(nn.Module):

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
        super(LateFusion, self).__init__()

        self.option = option

        self.image_model = ModelMap[image_model.lower()](hidden_dim, num_labels, model_name, finetune)
        self.text_model = ModelMap[text_model.lower()](word_dim, hidden_dim, num_labels, vocab_size, model_name)

        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):

        img_feature, img_predict = self.image_model(batch)
        txt_featrue, txt_predict = self.text_model(batch)

        if self.option == "mean":
            predict = torch.cat([img_predict.unsqueeze(0), txt_predict.unsqueeze(0)], 0)
            predict = torch.mean(predict, 0)
        else:
            temp = torch.cat([img_predict.unsqueeze(0), txt_predict.unsqueeze(0)], 0)
            sel = torch.abs(temp-0.5)
            sel_idx = torch.argmax(sel, dim=0)
            predict = (1-sel_idx) * img_predict + sel_idx * txt_predict

        predict = self.sigmoid(predict)
        loss = self.criterion(predict, batch['label'])

        return {
            'loss': loss,
            'logit': predict,
            'img_feature': img_feature,
            'txt_feature': txt_featrue
        }

