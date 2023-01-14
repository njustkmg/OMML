import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torchmm.models.fusion.backbone import Vgg, Resnet, Lstm, Gru
from torch.nn.init import xavier_normal

ModelMap = {
    'vgg': Vgg,
    'resnet': Resnet,
    'lstm': Lstm,
    'gru': Gru
}


class LMFFusion(nn.Module):

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

        self.image_factor = Parameter(torch.Tensor(self.rank, hidden_dim + 1, num_labels))
        self.text_factor = Parameter(torch.Tensor(self.rank, hidden_dim + 1, num_labels))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, num_labels))

        xavier_normal(self.image_factor)
        xavier_normal(self.text_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):

        DTYPE = torch.cuda.FloatTensor
        img_feature = self.image_model(batch)
        txt_feature = self.text_model(batch)
        batch_size = txt_feature.data.shape[0]

        img_feature = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), img_feature), dim=1)
        txt_feature = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), txt_feature), dim=1)

        fusion_img = torch.matmul(img_feature, self.image_factor)
        fusion_txt = torch.matmul(txt_feature, self.text_factor)
        fusion_zy = fusion_img * fusion_txt

        predict = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        predict = predict.view(-1, self.num_labels)

        predict = self.sigmoid(predict)
        loss = self.criterion(predict, batch['label'])

        return {
            'loss': loss,
            'logit': predict,
            'img_feature': img_feature,
            'txt_feature': txt_feature
        }
