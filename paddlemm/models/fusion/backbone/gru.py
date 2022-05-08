import paddle
import paddle.nn as nn


class Gru(nn.Layer):

    def __init__(self, word_dim, hidden_dim, num_labels, vocab_size, model_name):
        super(Gru, self).__init__()

        self.model_name = model_name.lower()
        self.input_dim = word_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        self.emb = nn.Embedding(vocab_size, self.input_dim)
        self.hidden = nn.GRU(self.input_dim, self.hidden_dim, 2)
        self.predict = nn.Linear(self.hidden_dim, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):

        txt_emb = self.emb(batch['text_token'])
        y, txt_hidden = self.hidden(txt_emb)
        txt_hidden = paddle.mean(txt_hidden, 0)
        txt_predict = self.predict(txt_hidden)

        if self.model_name == "earlyfusion" or self.model_name == "lmffusion":
            return txt_hidden
        elif self.model_name == "latefusion" or self.model_name == "tmcfusion":
            return txt_predict
        else:
            txt_predict = self.sigmoid(txt_predict)
            loss = self.criterion(txt_predict, batch['label'])
            if self.training:
                return loss
            else:
                return loss, txt_predict

