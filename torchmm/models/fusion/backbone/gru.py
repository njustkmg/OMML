import torch
import torch.nn as nn


class Gru(nn.Module):

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

        if torch.cuda.is_available():
            batch['text_token'] = batch['text_token'].cuda()
            batch['label'] = batch['label'].cuda()

        txt_emb = self.emb(batch['text_token'])  # [16, 59, 200]
        txt_hidden, y = self.hidden(txt_emb)  # [2, 16, 128]
        txt_hidden = torch.mean(txt_hidden, 1)
        txt_predict = self.predict(txt_hidden)

        if self.model_name == "earlyfusion" or self.model_name == "lmffusion":
            return txt_hidden
        elif self.model_name == "latefusion" or self.model_name == "tmcfusion":
            return txt_hidden, txt_predict
        else:
            txt_predict = self.sigmoid(txt_predict)
            loss = self.criterion(txt_predict, batch['label'])
            if self.training:
                return loss
            else:
                return loss, txt_predict

