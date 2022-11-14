import paddle
import paddle.nn as nn

from .utils import l2norm


def EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, text_norm=True, dropout=0.0):
    """A wrapper to text encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    model_name = model_name.lower()
    EncoderMap = {
        'scan': EncoderTextRegion,
        'vsepp': EncoderTextGlobal,
        'sgraf': EncoderTextRegion,
        'imram': EncoderTextRegion
    }

    if model_name in EncoderMap:
        txt_enc = EncoderMap[model_name](vocab_size, word_dim, embed_size, num_layers, use_bi_gru, text_norm, dropout)
    else:
        raise ValueError("Unknown model: {}".format(model_name))

    return txt_enc


class EncoderTextRegion(nn.Layer):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, text_norm=True, dropout=0.0):
        super(EncoderTextRegion, self).__init__()
        self.embed_size = embed_size
        self.text_norm = text_norm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim, weight_attr=nn.initializer.Uniform(low=-0.1, high=0.1))

        # caption embedding
        self.use_bi_gru = 'bidirectional' if use_bi_gru else 'forward'
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, direction=self.use_bi_gru)

        if dropout:
            self.use_drop = True
            self.dropout = nn.Dropout(dropout)
        else:
            self.use_drop = False

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        if self.use_drop:
            x = self.dropout(x)

        # Forward propagate RNN
        out, _ = self.rnn(x, None, lengths)

        if self.use_bi_gru == 'bidirectional':
            out = (out[:, :, :int(out.shape[2] / 2)] + out[:, :, int(out.shape[2] / 2):]) / 2

        # normalization in the joint embedding space
        if self.text_norm:
            out = l2norm(out, dim=-1)

        return out


class EncoderTextGlobal(nn.Layer):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, text_norm=True, dropout=0.0):
        super(EncoderTextGlobal, self).__init__()
        self.embed_size = embed_size
        self.text_norm = text_norm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim, weight_attr=nn.initializer.Uniform(low=-0.1, high=0.1))

        # caption embedding
        self.use_bi_gru = 'bidirectional' if use_bi_gru else 'forward'
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, direction=self.use_bi_gru)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate RNN
        out, _ = self.rnn(x, None, lengths)

        if self.use_bi_gru == 'bidirectional':
            out = (out[:, :, :int(out.shape[2] / 2)] + out[:, :, int(out.shape[2] / 2):]) / 2

        new_out = []
        for i in range(len(out)):
            new_out.append(out[i][lengths[i]-1])
        out = paddle.stack(new_out)

        # normalization in the joint embedding space
        if self.text_norm:
            out = l2norm(out, dim=-1)

        return out