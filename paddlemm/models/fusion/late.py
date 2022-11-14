import paddle
import paddle.nn as nn
from paddlemm.models.fusion.backbone import Vgg, Resnet, Lstm, Gru

ModelMap = {
    'vgg': Vgg,
    'resnet': Resnet,
    'lstm': Lstm,
    'gru': Gru
}

class LateFusion(nn.Layer):

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

        label = batch['label']

        img_predict = self.image_model(batch)
        txt_predict = self.text_model(batch)

        if self.option == "mean":
            predict = paddle.concat([img_predict.unsqueeze(0), txt_predict.unsqueeze(0)], axis=0)
            predict = paddle.mean(predict, 0)
        else:
            temp = paddle.concat([img_predict.unsqueeze(0), txt_predict.unsqueeze(0)], axis=0)
            sel = paddle.abs(temp - 0.5)
            sel_idx = paddle.argmax(sel, axis=0)
            sel_idx = paddle.to_tensor(sel_idx, dtype="float32")
            predict = (1 - sel_idx) * img_predict + sel_idx * txt_predict

        predict = self.sigmoid(predict)
        loss = self.criterion(predict, label)

        if self.training:
            return loss
        else:
            return loss, predict
