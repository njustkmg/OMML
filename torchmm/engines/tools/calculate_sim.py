"""Calculate the similarity between image and text for cross-modal retrieval tasks"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch

from torchmm.models import xattn_score_i2t, xattn_score_t2i


def calculate_sim(model, img_embs, cap_embs, cap_lens, img_fcs=None, cap_hts=None, **kwargs):
    cross_attn = kwargs.get("cross_attn", None)
    model_mode = kwargs.get("model_mode", None)
    model_name = kwargs.get('model_name').lower()
    iteration_step = kwargs.get('iteration_step', None)

    # start = time.time()
    if model_name == 'scan':
        if cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, **kwargs)
        elif cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, **kwargs)
        else:
            assert False, "wrong cross attn"
    elif model_name == 'sgraf':
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, **kwargs)
    elif model_name == 'imram':
        if model_mode == 'full_IMRAM':
            sims = shard_xattn_Full_IMRAM(model, img_fcs, img_embs, cap_hts, cap_embs, cap_lens, iteration_step, shard_size=32)
        elif model_mode == "image_IMRAM":
            sims = shard_xattn_Image_IMRAM(model, img_fcs, img_embs, cap_hts, cap_embs, cap_lens, iteration_step, shard_size=32)
        elif model_mode == "text_IMRAM":
            sims = shard_xattn_Text_IMRAM(model, img_fcs, img_embs, cap_hts, cap_embs, cap_lens, iteration_step, shard_size=32)
        else:
            assert False, "wrong model mode"
    else:
        sims = img_embs.dot(cap_embs.T)
    # end = time.time()
    # print("calculate similarity time:", end - start)

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


def shard_xattn_Full_IMRAM(model, images_fc, images, caption_ht, captions, caplens, iteration_step, shard_size=32):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d_t2i = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]
    d_i2t = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im_fc = torch.from_numpy(images_fc[im_start:im_end]).cuda()
            im_emb = torch.from_numpy(images[im_start:im_end]).cuda()
            h = torch.from_numpy(caption_ht[cap_start:cap_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim_list_t2i = model.xattn_score_Text_IMRAM(im_fc, im_emb, h, s, l)
            sim_list_i2t = model.xattn_score_Image_IMRAM(im_fc, im_emb, h, s, l)
            assert len(sim_list_t2i) == iteration_step and len(sim_list_i2t) == iteration_step
            for k in range(iteration_step):
                d_t2i[k][im_start:im_end, cap_start:cap_end] = sim_list_t2i[k].data.cpu().numpy()
                d_i2t[k][im_start:im_end, cap_start:cap_end] = sim_list_i2t[k].data.cpu().numpy()

    score = 0
    for j in range(iteration_step):
        score += d_t2i[j]
    for j in range(iteration_step):
        score += d_i2t[j]

    return score


def shard_xattn_Text_IMRAM(model, images_fc, images, caption_ht, captions, caplens, iteration_step, shard_size=32):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im_fc = torch.from_numpy(images_fc[im_start:im_end]).cuda()
            im_emb = torch.from_numpy(images[im_start:im_end]).cuda()
            h = torch.from_numpy(caption_ht[cap_start:cap_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim_list = model.xattn_score_Text_IMRAM(im_fc, im_emb, h, s, l)
            assert len(sim_list) == iteration_step
            for k in range(iteration_step):
                d[k][im_start:im_end, cap_start:cap_end] = sim_list[k].data.cpu().numpy()

    score = 0
    for j in range(iteration_step):
        score += d[j]

    return score


def shard_xattn_Image_IMRAM(model, images_fc, images, caption_ht, captions, caplens, iteration_step, shard_size=32):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im_fc = torch.from_numpy(images_fc[im_start:im_end]).cuda()
            im_emb = torch.from_numpy(images[im_start:im_end]).cuda()
            h = torch.from_numpy(caption_ht[cap_start:cap_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim_list = model.xattn_score_Image_IMRAM(im_fc, im_emb, h, s, l)
            assert len(sim_list) == iteration_step
            for k in range(iteration_step):
                if len(sim_list[k]) != 0:
                    d[k][im_start:im_end, cap_start:cap_end] = sim_list[k].data.cpu().numpy()

    score = 0
    for j in range(iteration_step):
        score += d[j]
    return score