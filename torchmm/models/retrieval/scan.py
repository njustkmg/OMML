import torch
import torch.nn as nn
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
        """Compute the image and caption embeddings
        """
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        cap_lens = lengths.tolist()

        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)

        return img_emb, cap_emb, cap_lens

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        cap_lens = lengths.tolist()

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
            shard_size = kwargs.get('shard_size', 1000)
            n_im_shard = int((len(images) - 1) / shard_size + 1)
            n_cap_shard = int((len(captions) - 1) / shard_size + 1)

            d = np.zeros((len(images), len(captions)))
            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im = torch.FloatTensor(images[im_start:im_end])
                    s = torch.FloatTensor(captions[cap_start:cap_end])
                    l = caplens[cap_start:cap_end].tolist()

                    if torch.cuda.is_available():
                        im = im.cuda()
                        s = s.cuda()

                    sim = xattn_score_t2i(im, s, l, **kwargs)
                    d[im_start:im_end, cap_start:cap_end] = sim.cpu().detach().numpy()
            return d

        def shard_xattn_i2t(images, captions, caplens, **kwargs):
            """
            Computer pairwise i2t image-caption distance with locality sharding
            """
            shard_size = kwargs.get('shard_size', 1000)
            n_im_shard = int((len(images) - 1) / shard_size + 1)
            n_cap_shard = int((len(captions) - 1) / shard_size + 1)

            d = np.zeros((len(images), len(captions)))
            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im = torch.FloatTensor(images[im_start:im_end])
                    s = torch.FloatTensor(captions[cap_start:cap_end])
                    l = caplens[cap_start:cap_end].tolist()

                    if torch.cuda.is_available():
                        im = im.cuda()
                        s = s.cuda()
                    sim = xattn_score_i2t(im, s, l, **kwargs)
                    d[im_start:im_end, cap_start:cap_end] = sim.cpu().detach().numpy()
            return d

        cross_attn = kwargs.get('cross_attn')

        if cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, **kwargs)
        elif cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, **kwargs)
        else:
            assert False, "wrong cross attn"

        return sims
