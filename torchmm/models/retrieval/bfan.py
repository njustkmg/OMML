import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F

from .layers.utils import l1norm, l2norm, cosine_similarity, cosine_similarity_a2a
from .layers.contrastive import ContrastiveLoss
from .layers.img_enc import EncoderImage
from .layers.txt_enc import EncoderText


def func_attention(query, context, focal_type, lambda_softmax, eps=1e-8):
    """
    query: (batch, queryL, d)
    context: (batch, sourceL, d)
    """
    batch_size, queryL, sourceL = context.size(
        0), query.size(1), context.size(1)

    # Step 1: preassign attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*lambda_softmax)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if focal_type == 'equal':
        funcH = focal_equal(attn, batch_size, queryL, sourceL)
    elif focal_type == 'prob':
        funcH = focal_prob(attn, batch_size, queryL, sourceL)
    else:
        raise ValueError("unknown focal attention type:", focal_type)

    # Step 3: reassign attention
    tmp_attn = funcH * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # --> (batch, sourceL, queryL)
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, re_attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext


def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def focal_prob(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as the sqrt
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """

    # -> (batch, queryL, sourceL, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj = attn.unsqueeze(2).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(batch_size*queryL, sourceL, 1)
    xj = xj.view(batch_size*queryL, 1, sourceL)
    xj_confi = xj_confi.view(batch_size*queryL, 1, sourceL)

    # -> (batch*queryL, sourceL, sourceL)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    funcF = torch.sum(term1-term2, dim=-1)  # -> (batch*queryL, sourceL)
    funcF = funcF.view(batch_size, queryL, sourceL)

    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def xattn_score(images, captions, cap_lens, focal_type, lambda_softmax):
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

        # Focal attention in text-to-image direction
        # weiContext: (n_image, n_word, d)
        weiContext = func_attention(cap_i_expand, images, focal_type, lambda_softmax)
        t2i_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        t2i_sim = t2i_sim.mean(dim=1, keepdim=True)

        # Focal attention in image-to-text direction
        # weiContext: (n_image, n_word, d)
        weiContext = func_attention(images, cap_i_expand, focal_type, lambda_softmax)
        i2t_sim = cosine_similarity(images, weiContext, dim=2)
        i2t_sim = i2t_sim.mean(dim=1, keepdim=True)

        # Overall similarity for image and text
        sim = t2i_sim + i2t_sim
        similarities.append(sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


class BFAN(nn.Module):
    """
    Bidirectional Focal Attention Network (BFAN) model
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
                 focal_type,
                 lambda_softmax,
                 use_bi_gru=True,
                 image_norm=True,
                 text_norm=True,
                 **kwargs
                 ):
        super(BFAN, self).__init__()
        # Build Models
        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers,
                                   use_bi_gru=use_bi_gru, text_norm=text_norm)

        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)
        self.focal_type = focal_type
        self.lambda_softmax = lambda_softmax

    def forward_emb(self, batch):
        """Compute the image and caption embeddings
        """
        images = batch['image_feat']
        captions = batch['text_token']
        cap_lens = batch['text_len']

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        cap_lens = cap_lens.tolist()

        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)

        return img_emb, cap_emb, cap_lens

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        cap_lens = batch['text_len']

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        cap_lens = cap_lens.tolist()

        # compute the embeddings
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)

        score = xattn_score(img_emb, cap_emb, cap_lens, self.focal_type, self.lambda_softmax)
        loss = self.criterion(score)

        return loss

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):
        shard_size = kwargs.get('shard_size', 128)
        focal_type = model.focal_type
        lambda_softmax = model.lambda_softmax

        n_im_shard = int((len(img_embs)-1)/shard_size + 1)
        n_cap_shard = int((len(cap_embs)-1)/shard_size + 1)

        sims = np.zeros((len(img_embs), len(cap_embs)))
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
            for j in range(n_cap_shard):
                cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

                with torch.no_grad():
                    im = torch.FloatTensor(img_embs[im_start:im_end])
                    ca = torch.FloatTensor(cap_embs[cap_start:cap_end])
                    l = cap_lens[cap_start:cap_end].tolist()

                    if torch.cuda.is_available():
                        im = im.cuda()
                        ca = ca.cuda()

                sim = xattn_score(im, ca, l, focal_type, lambda_softmax)
                sims[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()

        return sims