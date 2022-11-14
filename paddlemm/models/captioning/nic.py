import paddle
import paddle.nn as nn
from paddle.vision import models


class Encoder(nn.Layer):
    def __init__(self, network='resnet152'):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = models.resnet152(pretrained=True)
            # params = paddle.load('/test/whc/PaddleMM/resnet152.pdparams')
            # self.net.set_dict(params)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        else:
            self.net = models.vgg16(pretrained=True)
            # params = paddle.load('/test/whc/PaddleMM/vgg16.pdparams')
            # self.net.set_dict(params)
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 512

        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.net(x)
        x = paddle.transpose(x, (0, 2, 3, 1))
        x = paddle.reshape(x, [x.shape[0], -1, x.shape[-1]])
        return x


class Attention(nn.Layer):
    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(axis=1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(axis=1)
        return context, alpha


class Decoder(nn.Layer):

    def __init__(self, vocab_size, encoder_dim, tf=False):
        super(Decoder, self).__init__()
        self.use_tf = tf

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, 512)
        self.init_c = nn.Linear(encoder_dim, 512)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(512, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(512, vocab_size)
        self.dropout = nn.Dropout()

        self.attention = Attention(encoder_dim)
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTMCell(512 + encoder_dim, 512)

    def forward(self, img_features, captions, caption_lengths):

        batch_size = img_features.shape[0]

        h, c = self.get_init_lstm_state(img_features)
        # decode_lengths = [int(c-1) for c in caption_lengths]
        # max_timespan = max(decode_lengths)
        max_timespan = captions.shape[1]-1

        # 1 is <start>
        prev_words = paddle.ones([batch_size, 1], dtype='int64')
        if self.use_tf:
            embedding = self.embedding(captions) if self.training else self.embedding(prev_words)
        else:
            embedding = self.embedding(prev_words)

        preds = []
        alphas = []
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            if self.use_tf and self.training:
                lstm_input = paddle.concat((embedding[:, t], gated_context), axis=1)
            else:
                embedding = embedding.squeeze(1) if len(embedding.shape) == 3 else embedding
                lstm_input = paddle.concat((embedding, gated_context), axis=1)

            h, (_, c) = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            preds.append(output.unsqueeze(1))
            alphas.append(alpha.unsqueeze(1))

            if not self.training or not self.use_tf:
                embedding = self.embedding(paddle.argmax(output, -1))

        preds = paddle.concat(preds, 1)
        alphas = paddle.concat(alphas, 1)
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(axis=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c


class NIC(nn.Layer):

    def __init__(self,
                 network,
                 vocab_size,
                 teacher_forcing,
                 alpha_c,
                 **kwargs):

        super(NIC, self).__init__()

        self.encoder = Encoder(network=network)
        self.decoder = Decoder(vocab_size=vocab_size,
                               encoder_dim=self.encoder.dim,
                               tf=teacher_forcing)
        self.criterion = nn.CrossEntropyLoss()
        self.alpha_c = alpha_c

    def forward(self, batch):

        images = batch['image_feat']
        captions = batch['text_token']
        caplens = batch['text_len']

        # with paddle.no_grad():
        images = self.encoder(images)
        logit, alphas = self.decoder(images, captions, caplens)
        target = captions[:, 1:captions.shape[1]]

        scores = paddle.reshape(logit, [-1, logit.shape[-1]])
        target = paddle.reshape(target, [-1, 1]).squeeze()

        # gt = []
        # pd = []
        # for i in range(len(caplens)):
        #     gt.append(target[i, 1:caplens[i]-1])
        #     pd.append(logit[i, 1:caplens[i]-1, :])
        # gt = paddle.concat(gt, axis=0)
        # pd = paddle.concat(pd, axis=0)
        # loss = self.criterion(pd, gt)

        loss = self.criterion(scores, target)
        loss += self.alpha_c * ((1. - alphas.sum(axis=1)) ** 2).mean()

        if self.training:
            return loss
        else:
            logit = paddle.argmax(logit, axis=-1)
            return loss, logit
