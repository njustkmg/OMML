import paddle
import paddle.nn as nn
import numpy as np

from .layers.contrastive import ContrastiveLoss
from .layers.utils import l1norm, l2norm, cosine_similarity
from .layers.img_enc import EncoderImage
from .layers.txt_enc import EncoderText


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
    attn = nn.Softmax(axis=1)(attn * smooth)
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
        cap_i_expand = paddle.expand(cap_i, [n_image, cap_i.shape[1], cap_i.shape[2]])
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, raw_feature_norm, smooth=lambda_softmax)
        # print(weiContext)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
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


class SCAN(nn.Layer):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(self,
                 model_name,
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
                 image_norm=True,
                 text_norm=True,
                 **kwargs):

        super(SCAN, self).__init__()
        # Build Models
        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers,
                                   use_bi_gru=use_bi_gru, text_norm=text_norm)

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
        cap_lens = batch['text_len']

        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)

        return img_emb, cap_emb, cap_lens

    def forward(self, batch):

        images = batch['image_feat']
        captions = batch['text_token']
        cap_lens = batch['text_len']

        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)

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

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):

        def shard_xattn_t2i(images, captions, caplens, **kwargs):
            """
            Computer pairwise t2i image-caption distance with locality sharding
            """
            shard_size = kwargs.get('shard_size', 128)
            n_im_shard = int((len(images) - 1) / shard_size + 1)
            n_cap_shard = int((len(captions) - 1) / shard_size + 1)

            d = np.zeros((len(images), len(captions)))
            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im = paddle.to_tensor(images[im_start:im_end])
                    s = paddle.to_tensor(captions[cap_start:cap_end])
                    l = paddle.to_tensor(caplens[cap_start:cap_end])
                    sim = xattn_score_t2i(im, s, l, **kwargs)
                    d[im_start:im_end, cap_start:cap_end] = np.array(sim)
            return d

        def shard_xattn_i2t(images, captions, caplens, **kwargs):
            """
            Computer pairwise i2t image-caption distance with locality sharding
            """
            shard_size = kwargs.get('shard_size', 128)
            n_im_shard = int((len(images) - 1) / shard_size + 1)
            n_cap_shard = int((len(captions) - 1) / shard_size + 1)

            d = np.zeros((len(images), len(captions)))
            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im = paddle.to_tensor(images[im_start:im_end])
                    s = paddle.to_tensor(captions[cap_start:cap_end])
                    l = paddle.to_tensor(caplens[cap_start:cap_end])
                    sim = xattn_score_i2t(im, s, l, **kwargs)
                    d[im_start:im_end, cap_start:cap_end] = np.array(sim)
            return d

        cross_attn = kwargs.get('cross_attn')

        if cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, **kwargs)
        elif cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, **kwargs)
        else:
            assert False, "wrong cross attn"

        return sims