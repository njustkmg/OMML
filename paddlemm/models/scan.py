import numpy as np

import paddle
import paddle.nn as nn
from paddle.nn.utils.weight_norm_hook import weight_norm

from .layers.normalize import l1norm, l2norm, cosine_similarity
from .layers.contrastive import ContrastiveLoss


def EncoderImage(img_dim, embed_size, precomp_enc_type='basic', image_norm=True):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(img_dim, embed_size, image_norm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(img_dim, embed_size, image_norm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Layer):

    def __init__(self, img_dim, embed_size, image_norm=True):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.image_norm = image_norm
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.weight.shape[0] + self.fc.weight.shape[1])
        v = np.random.uniform(-r, r, size=(self.fc.weight.shape[0], self.fc.weight.shape[1])).astype('float32')
        b = np.zeros(self.fc.bias.shape).astype('float32')
        self.fc.weight.set_value(v)
        self.fc.bias.set_value(b)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if self.image_norm:
            features = l2norm(features, dim=-1)

        return features


class EncoderImageWeightNormPrecomp(nn.Layer):

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
class EncoderText(nn.Layer):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, image_norm=True):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.image_norm = image_norm

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

        # normalization in the joint embedding space
        if self.image_norm:
            out = l2norm(out, dim=-1)

        return out, lengths


def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.shape[0], query.shape[1]
    batch_size, sourceL = context.shape[0], context.shape[1]

    # Get attention
    # --> (batch, d, queryL)
    queryT = paddle.transpose(query, (0, 2, 1))

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = paddle.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = paddle.reshape(attn, [batch_size * sourceL, queryL])
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = paddle.reshape(attn, [batch_size, sourceL, queryL])
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        # attn = nn.ReLU()(attn)+0.1
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
    attn = paddle.transpose(attn, (0, 2, 1))
    # --> (batch*queryL, sourceL)
    attn = paddle.reshape(attn, [batch_size * queryL, sourceL])
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = paddle.reshape(attn, [batch_size, queryL, sourceL])
    # --> (batch, sourceL, queryL)
    attnT = paddle.transpose(attn, (0, 2, 1))
    # --> (batch, d, sourceL)
    contextT = paddle.transpose(context, (0, 2, 1))
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = paddle.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = paddle.transpose(weightedContext, (0, 2, 1))

    return weightedContext, attnT


def xattn_score_t2i(images, captions, cap_lens, raw_feature_norm, agg_func, lambda_lse, lambda_softmax, **kwargs):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.shape[0]
    n_caption = captions.shape[0]
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0)
        # --> (n_image, n_word, d)
        # cap_i_expand = paddle.concat([cap_i for _ in range(n_image)], axis=0)
        cap_i_expand = paddle.expand(cap_i, [n_image, cap_i.shape[1], cap_i.shape[2]])
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, raw_feature_norm, smooth=lambda_softmax)
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if agg_func == 'LogSumExp':
            row_sim = paddle.exp(row_sim * lambda_lse)
            row_sim = row_sim.sum(axis=1, keepdim=True)
            row_sim = paddle.log(row_sim) / lambda_lse
        elif agg_func == 'Max':
            row_sim = row_sim.max(axis=1, keepdim=True)
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(axis=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(axis=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = paddle.concat(similarities, 1)

    return similarities


def xattn_score_i2t(images, captions, cap_lens, raw_feature_norm, agg_func, lambda_lse, lambda_softmax, **kwargs):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.shape[0]
    n_caption = captions.shape[0]
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0)
        # (n_image, n_word, d)
        cap_i_expand = paddle.concat([cap_i for _ in range(n_image)], axis=0)
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
            row_sim = paddle.exp(row_sim * lambda_lse)
            row_sim = row_sim.sum(axis=1, keepdim=True)
            row_sim = paddle.log(row_sim) / lambda_lse
        elif agg_func == 'Max':
            row_sim = row_sim.max(axis=1, keepdim=True)[0]
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(axis=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(axis=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = paddle.concat(similarities, 1)
    return similarities


class SCAN(nn.Layer):
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
                                   use_bi_gru=use_bi_gru, image_norm=text_norm)

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)

        self.cross_attn = cross_attn
        self.raw_feature_norm = raw_feature_norm
        self.agg_func = agg_func
        self.lambda_lse = lambda_lse
        self.lambda_softmax = lambda_softmax

    def forward_emb(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        img_emb = self.img_enc(images)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)

        return img_emb, cap_emb, cap_lens

    def forward(self, batch):

        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

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
        # print(scores)
        loss = self.criterion(scores)

        return loss
