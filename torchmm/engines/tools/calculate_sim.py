"""Calculate the similarity between image and text for cross-modal retrieval tasks"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch

from torchmm.models import xattn_score_i2t, xattn_score_t2i


def calculate_sim(model, img_embs, cap_embs, cap_lens, **kwargs):
    cross_attn = kwargs.get("cross_attn", None)

    start = time.time()
    if cross_attn == 't2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, **kwargs)
    elif cross_attn == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, **kwargs)
    else:
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, **kwargs)
    end = time.time()
    print("calculate similarity time:", end - start)

    return sims


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
            cap_start, cap_end = shard_size * j, min(shard_size *(j + 1), len(captions))
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
    shard_size = kwargs.get('shard_size', 128)
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


def shard_attn_scores(model, img_embs, cap_embs, cap_lens, **kwargs):
    shard_size = kwargs.get('shard_size', 100)

    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.FloatTensor(img_embs[im_start:im_end])
                ca = torch.FloatTensor(cap_embs[ca_start:ca_end])
                l = cap_lens[ca_start:ca_end].tolist()

                if torch.cuda.is_available():
                    im = im.cuda()
                    ca = ca.cuda()

                sim = model.forward_sim((im, ca, l))

            sims[im_start:im_end, ca_start:ca_end] = sim.cpu().detach().numpy()
    return sims