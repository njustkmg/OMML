import torch
import torch.nn as nn
from torchvision.models import resnet152, vgg16


class Encoder(nn.Module):
    def __init__(self, network='vgg16'):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        else:
            self.net = vgg16(pretrained=True)
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 512

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x


class Attention(nn.Module):
    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha


class Decoder(nn.Module):
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

    def forward(self, img_features, captions):
        """
        We can use teacher forcing during training. For reference, refer to
        https://www.deeplearningbook.org/contents/rnn.html
        """
        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features)
        # max_timespan = max([len(caption) for caption in captions]) - 1
        max_timespan = captions.size(1) - 1

        prev_words = torch.ones(batch_size, 1).long()
        if torch.cuda.is_available():
            prev_words = prev_words.cuda()

        if self.use_tf:
            embedding = self.embedding(captions) if self.training else self.embedding(prev_words)
        else:
            embedding = self.embedding(prev_words)

        # preds = torch.zeros(batch_size, max_timespan, self.vocab_size)
        # alphas = torch.zeros(batch_size, max_timespan, img_features.size(1))
        # if torch.cuda.is_available():
        #     preds = preds.cuda()
        #     alphas = alphas.cuda()

        preds = []
        alphas = []

        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            # preds[:, t] = output
            # alphas[:, t] = alpha

            preds.append(output.unsqueeze(1))
            alphas.append(alpha.unsqueeze(1))

            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))

        preds = torch.cat(preds, 1)
        alphas = torch.cat(alphas, 1)

        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c


class NIC(nn.Module):

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

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        with torch.no_grad():
            images = self.encoder(images)
        logit, alphas = self.decoder(images, captions)
        target = captions[:, 1:captions.size(1)]

        scores = logit.reshape([-1, logit.size(-1)])
        target = target.reshape([-1, 1]).squeeze()

        loss = self.criterion(scores, target)
        loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        if self.training:
            return loss
        else:
            logit = torch.argmax(logit, dim=-1)
            return loss, logit