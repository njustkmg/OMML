import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import XavierNormal, Assign
from paddlemm.models.fusion.backbone import Vgg, Resnet, Lstm, Gru


ModelMap = {
    'vgg': Vgg,
    'resnet': Resnet,
    'lstm': Lstm,
    'gru': Gru
}


class LMFFusion(nn.Layer):

    def __init__(self,
                 image_model,
                 text_model,
                 word_dim,
                 vocab_size,
                 hidden_dim,
                 num_labels,
                 model_name,
                 finetune,
                 **kwargs):
        super(LMFFusion, self).__init__()

        # self.batch_size = kwargs['batch_size']
        self.rank = kwargs['rank']
        self.num_labels = num_labels
        self.image_model = ModelMap[image_model.lower()](hidden_dim, num_labels, model_name, finetune)
        self.text_model = ModelMap[text_model.lower()](word_dim, hidden_dim, num_labels, vocab_size, model_name)

        self.image_factor = paddle.create_parameter([self.rank, hidden_dim + 1, self.num_labels], dtype='float32',
                                                    default_initializer=XavierNormal())
        self.text_factor = paddle.create_parameter([self.rank, hidden_dim + 1, self.num_labels], dtype='float32',
                                                   default_initializer=XavierNormal())
        self.fusion_weights = paddle.create_parameter([1, self.rank], dtype='float32',
                                                      default_initializer=XavierNormal())
        self.fusion_bias = paddle.create_parameter([1, self.num_labels], dtype='float32',
                                                   default_initializer=Assign(np.zeros([1, self.num_labels]).astype('float32')))


        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):

        img_predict = self.image_model(batch)
        txt_predict = self.text_model(batch)
        batch_size = txt_predict.shape[0]
        img_predict = paddle.concat((paddle.to_tensor(paddle.ones([batch_size, 1]), dtype='float32'), img_predict), axis=1)
        txt_predict = paddle.concat((paddle.to_tensor(paddle.ones([batch_size, 1]), dtype='float32'), txt_predict), axis=1)

        fusion_img = paddle.matmul(img_predict, self.image_factor)
        fusion_txt = paddle.matmul(txt_predict, self.text_factor)
        fusion_zy = fusion_img * fusion_txt

        predict = paddle.matmul(self.fusion_weights, paddle.transpose(fusion_zy, perm=[1,0,2])).squeeze() + self.fusion_bias

        predict = self.sigmoid(predict)
        loss = self.criterion(predict, batch['label'])

        if self.training:
            return loss
        else:
            return loss, predict
