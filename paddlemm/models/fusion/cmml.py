import paddle
import paddle.nn as nn
from paddle.vision import models


class CMML(nn.Layer):
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
        self.softmax = nn.Softmax(axis=1)

        # Loss function
        self.criterion = nn.BCELoss()
        self.cita = 1.003

    def pretrain(self, batch):
        feature, label = batch['feature'], batch['label']
        supervised_txt, supervised_img, unsupervised_txt, unsupervised_img = feature
        supervised_txt = paddle.concat(supervised_txt, 0)
        supervised_img = paddle.concat(supervised_img, 0)
        label = paddle.concat(label, 0)

        supervised_img_hidden = self.resnet(supervised_img)
        supervised_img_hidden = paddle.reshape(supervised_img_hidden, shape=[supervised_img_hidden.shape[0], 512])
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
        supervised_txt = paddle.concat(supervised_txt, 0)
        supervised_img = paddle.concat(supervised_img, 0)
        unsupervised_txt = paddle.concat(unsupervised_txt, 0)
        unsupervised_img = paddle.concat(unsupervised_img, 0)
        label = paddle.concat(label, 0)

        # supervise training
        supervised_txt_hidden = self.txt_hidden(supervised_txt)
        supervised_txt_predict = self.txt_predict(supervised_txt_hidden)
        supervised_txt_predict = self.sigmoid(supervised_txt_predict)

        supervised_img_hidden = self.resnet(supervised_img)
        supervised_img_hidden = paddle.reshape(supervised_img_hidden, shape=[supervised_img_hidden.shape[0], 512])
        supervised_img_hidden = self.img_hidden(supervised_img_hidden)
        supervised_img_predict = self.img_predict(supervised_img_hidden)
        supervised_img_predict = self.sigmoid(supervised_img_predict)

        attn_txt = self.attn_mlp(supervised_txt_hidden)
        attn_img = self.attn_mlp(supervised_img_hidden)
        attn_modality = paddle.concat([attn_txt, attn_img], axis=1)
        attn_modality = self.softmax(attn_modality)
        attn_img = paddle.zeros(shape=[1, len(label)])
        attn_img[0] = attn_modality[:, 0]
        attn_img = paddle.t(attn_img)
        attn_txt = paddle.zeros(shape=[1, len(label)])
        attn_txt[0] = attn_modality[:, 1]
        attn_txt = paddle.t(attn_txt)

        supervised_hidden = attn_txt * supervised_txt_hidden + attn_img * supervised_img_hidden
        supervised_predict = self.modality_predict(supervised_hidden)
        supervised_predict = self.sigmoid(supervised_predict)

        mm_loss = self.criterion(supervised_predict, label)
        txt_loss = self.criterion(supervised_txt_predict, label)
        img_loss = self.criterion(supervised_img_predict, label)
        supervised_loss = mm_loss * 3 + txt_loss + img_loss

        # diversity measure
        similar = paddle.bmm(supervised_img_predict.unsqueeze(1),
                             supervised_txt_predict.unsqueeze(2))

        similar = paddle.reshape(similar, shape=[supervised_img_predict.shape[0]])
        norm_matrix_img = paddle.norm(supervised_img_predict, p=2, axis=1)
        norm_matrix_text = paddle.norm(supervised_txt_predict, p=2, axis=1)
        div = paddle.mean(similar / (norm_matrix_img * norm_matrix_text))

        # unsupervise training
        # Robust Consistency Measure

        unsupervised_txt_hidden = self.txt_hidden(unsupervised_txt)
        unsupervised_txt_predict = self.txt_predict(unsupervised_txt_hidden)
        unsupervised_txt_predict = self.sigmoid(unsupervised_txt_predict)

        unsupervised_img_hidden = self.resnet(unsupervised_img)
        unsupervised_img_hidden = paddle.reshape(unsupervised_img_hidden, shape=[unsupervised_img_hidden.shape[0], 512])
        unsupervised_img_hidden = self.img_hidden(unsupervised_img_hidden)
        unsupervised_img_predict = self.img_predict(unsupervised_img_hidden)
        unsupervised_img_predict = self.sigmoid(unsupervised_img_predict)

        unsimilar = paddle.bmm(unsupervised_img_predict.unsqueeze(1),
                               unsupervised_txt_predict.unsqueeze(2))

        unsimilar = paddle.reshape(unsimilar, shape=[unsupervised_img_predict.shape[0]])

        unnorm_matrix_img = paddle.norm(unsupervised_img_predict, p=2, axis=1)
        unnorm_matrix_text = paddle.norm(unsupervised_txt_predict, p=2, axis=1)

        dis = 2 - unsimilar / (unnorm_matrix_img * unnorm_matrix_text)

        # dis = paddle.abs(dis)
        # dis2 = paddle.sort(dis, axis=0)
        # split = int(np.sum((dis2<self.cita).numpy()))+1
        # tensor12 = dis[:split]
        # tensor22 = dis[split:]

        mask_1 = paddle.abs(dis) < self.cita
        tensor1 = paddle.masked_select(dis, mask_1)
        mask_2 = paddle.abs(dis) >= self.cita
        tensor2 = paddle.masked_select(dis, mask_2)
        tensor1loss = paddle.sum(tensor1 * tensor1 / 2)
        tensor2loss = paddle.sum(self.cita * (paddle.abs(tensor2) - 1 / 2 * self.cita))

        unsupervised_loss = (tensor1loss + tensor2loss) / unsupervised_img.shape[0]
        total_loss = supervised_loss + 0.01 * div + unsupervised_loss

        return total_loss

    def forward_eval(self, batch):
        """for evaluation, batch only contain supervised data"""

        feature, label = batch['feature'], batch['label']
        supervised_txt, supervised_img = feature

        supervised_txt_hidden = self.txt_hidden(supervised_txt)
        supervised_img_hidden = self.resnet(supervised_img)
        supervised_img_hidden = paddle.reshape(supervised_img_hidden, shape=[supervised_img_hidden.shape[0], 512])
        supervised_img_hidden = self.img_hidden(supervised_img_hidden)

        attn_txt = self.attn_mlp(supervised_txt_hidden)
        attn_img = self.attn_mlp(supervised_img_hidden)
        attn_modality = paddle.concat([attn_txt, attn_img], axis=1)
        attn_modality = self.softmax(attn_modality)
        attn_img = paddle.zeros(shape=[1, len(label)])
        attn_img[0] = attn_modality[:, 0]
        attn_img = paddle.t(attn_img)
        attn_txt = paddle.zeros(shape=[1, len(label)])
        attn_txt[0] = attn_modality[:, 1]
        attn_txt = paddle.t(attn_txt)

        supervised_hidden = attn_txt * supervised_txt_hidden + attn_img * supervised_img_hidden
        supervised_predict = self.modality_predict(supervised_hidden)
        supervised_predict = self.sigmoid(supervised_predict)

        total_loss = self.criterion(supervised_predict, label)

        return total_loss, supervised_predict