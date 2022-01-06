import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import numpy as np

from .layers.contrastive import ContrastiveLoss
from .layers.normalize import l1norm, l2norm, cosine_similarity


def EncoderImage(img_dim, embed_size, precomp_enc_type='basic',
                 image_norm=True):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, image_norm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, image_norm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, image_norm=True):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.image_norm = image_norm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if self.image_norm:
            features = l2norm(features, dim=-1)

        return features


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, image_norm=True):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.image_norm = image_norm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if self.image_norm:
            features = l2norm(features, dim=-1)

        return features


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, text_norm=True):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.text_norm = text_norm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
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
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        # normalization in the joint embedding space
        if self.text_norm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def xattn_score_t2i(images, captions, cap_lens, raw_feature_norm, agg_func, lambda_lse, lambda_softmax, **kwargs):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, raw_feature_norm, smooth=lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if agg_func == 'LogSumExp':
            row_sim.mul_(lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / lambda_lse
        elif agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t(images, captions, cap_lens, raw_feature_norm, agg_func, lambda_lse, lambda_softmax, **kwargs):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, raw_feature_norm, smooth=lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if agg_func == 'LogSumExp':
            row_sim.mul_(lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / lambda_lse
        elif agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class SCAN(nn.Module):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(self,
                 embed_size,
                 vocab_size,
                 word_dim,
                 num_layers,
                 image_dim,
                 margin,
                 max_violation,
                 cross_attn,
                 raw_feature_norm,
                 agg_func,
                 lambda_lse,
                 lambda_softmax,
                 use_bi_gru=True,
                 enc_type='basic',
                 image_norm=True,
                 text_norm=True,
                 **kwargs):

        super(SCAN, self).__init__()
        # Build Models
        self.img_enc = EncoderImage(image_dim, embed_size,
                                    precomp_enc_type=enc_type, image_norm=image_norm)
        self.txt_enc = EncoderText(vocab_size, word_dim, embed_size, num_layers,
                                   use_bi_gru=use_bi_gru, text_norm=text_norm)

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)

        self.cross_attn = cross_attn
        self.raw_feature_norm = raw_feature_norm
        self.agg_func = agg_func
        self.lambda_lse = lambda_lse
        self.lambda_softmax = lambda_softmax

    def forward_emb(self, batch):
        """Compute the image and caption embeddings
        """
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        lengths = lengths.tolist()

        img_emb = self.img_enc(images)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)

        return img_emb, cap_emb, cap_lens

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        lengths = lengths.tolist()

        img_emb = self.img_enc(images)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)

        if self.cross_attn == 't2i':
            scores = xattn_score_t2i(img_emb, cap_emb, cap_lens,
                                     raw_feature_norm=self.raw_feature_norm,
                                     agg_func=self.agg_func,
                                     lambda_lse=self.lambda_lse,
                                     lambda_softmax=self.lambda_softmax)
        elif self.cross_attn == 'i2t':
            scores = xattn_score_i2t(img_emb, cap_emb, cap_lens,
                                     raw_feature_norm=self.raw_feature_norm,
                                     agg_func=self.agg_func,
                                     lambda_lse=self.lambda_lse,
                                     lambda_softmax=self.lambda_softmax)
        else:
            raise ValueError("unknown attention type")

        loss = self.criterion(scores)

        return loss

