import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

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
        'imram': EncoderTextRegion,
        'bfan': EncoderTextRegion,
        'pfan': EncoderTextRegion
    }

    if model_name in EncoderMap:
        txt_enc = EncoderMap[model_name](vocab_size, word_dim, embed_size, num_layers, use_bi_gru, text_norm, dropout)
    else:
        raise ValueError("Unknown model: {}".format(model_name))

    return txt_enc


class EncoderTextRegion(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, text_norm=True, dropout=0.0):

        super(EncoderTextRegion, self).__init__()
        self.embed_size = embed_size
        self.text_norm = text_norm
        self.use_bi_gru = use_bi_gru

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        if dropout:
            self.use_drop = True
            self.dropout = nn.Dropout(dropout)
        else:
            self.use_drop = False

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        if self.use_drop:
            x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        # normalization in the joint embedding space
        if self.text_norm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


class EncoderTextGlobal(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, text_norm=True, dropout=0.0):

        super(EncoderTextGlobal, self).__init__()
        self.embed_size = embed_size
        self.text_norm = text_norm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out, 1)

        # take absolute value, used by order embeddings
        if self.text_norm:
            out = l2norm(out, dim=-1)

        return out