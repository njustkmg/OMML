import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from .layers.utils import l1norm, l2norm, cosine_similarity, cosine_similarity_a2a
from .layers.contrastive import ContrastiveLoss
from .layers.img_enc import EncoderImage
from .layers.txt_enc import EncoderText


def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8, weight=None):
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

    if weight is not None:
        attn = attn + weight

    attn_out = attn.clone()

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

    return weightedContext, attn_out


class IMRAM(nn.Layer):

    def __init__(self,
                 model_name,
                 embed_size,
                 vocab_size,
                 word_dim,
                 num_layers,
                 image_dim,
                 margin,
                 max_violation,
                 model_mode,
                 raw_feature_norm,
                 agg_func,
                 lambda_lse,
                 lambda_softmax,
                 iteration_step,
                 use_bi_gru=True,
                 image_norm=True,
                 text_norm=True,
                 **kwargs):
        super(IMRAM, self).__init__()
        # Build Models
        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers,
                                   use_bi_gru=use_bi_gru, text_norm=text_norm)

        self.linear_t2i = nn.Linear(embed_size * 2, embed_size)
        self.gate_t2i = nn.Linear(embed_size * 2, embed_size)
        self.linear_i2t = nn.Linear(embed_size * 2, embed_size)
        self.gate_i2t = nn.Linear(embed_size * 2, embed_size)

        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)
        self.model_mode = model_mode
        self.lambda_lse = lambda_lse
        self.lambda_softmax = lambda_softmax
        self.agg_func = agg_func
        self.iteration_step = iteration_step
        self.raw_feature_norm = raw_feature_norm

    def gated_memory_t2i(self, input_0, input_1):

        input_cat = paddle.concat([input_0, input_1], 2)
        input_1 = paddle.tanh(self.linear_t2i(input_cat))
        gate = F.sigmoid(self.gate_t2i(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output

    def gated_memory_i2t(self, input_0, input_1):

        input_cat = paddle.concat([input_0, input_1], 2)
        input_1 = paddle.tanh(self.linear_i2t(input_cat))
        gate = F.sigmoid(self.gate_i2t(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output

    def forward_emb(self, batch):
        """Compute the image and caption embeddings
        """
        images = batch['image_feat']
        captions = batch['text_token']
        cap_lens = batch['text_len']

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)

        return img_emb, cap_emb, cap_lens

    def forward(self, batch):

        images = batch['image_feat']
        captions = batch['text_token']
        cap_lens = batch['text_len']

        # compute the embeddings
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)

        if self.model_mode == "full_IMRAM":
            scores_t2i = self.xattn_score_Text_IMRAM(img_emb, cap_emb, cap_lens)
            scores_i2t = self.xattn_score_Image_IMRAM(img_emb, cap_emb, cap_lens)
            scores_t2i = paddle.stack(scores_t2i, 0).sum(0)
            scores_i2t = paddle.stack(scores_i2t, 0).sum(0)
            score = scores_t2i + scores_i2t
        elif self.model_mode == "image_IMRAM":
            scores_i2t = self.xattn_score_Image_IMRAM(img_emb, cap_emb, cap_lens)
            scores_i2t = paddle.stack(scores_i2t, 0).sum(0)
            score = scores_i2t
        elif self.model_mode == "text_IMRAM":
            scores_t2i = self.xattn_score_Text_IMRAM(img_emb, cap_emb, cap_lens)
            scores_t2i = paddle.stack(scores_t2i, 0).sum(0)
            score = scores_t2i
        else:
            raise ValueError('No such mode!')

        loss = self.criterion(score)
        return loss

    def xattn_score_Text_IMRAM(self, images, captions_all, cap_lens):
        """
        Images: (n_image, n_regions, d) matrix of images
        captions_all: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = [[] for _ in range(self.iteration_step)]
        n_image = images.shape[0]
        n_caption = captions_all.shape[0]
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0)
            # --> (n_image, n_word, d)
            cap_i_expand = paddle.expand(cap_i, [n_image, cap_i.shape[1], cap_i.shape[2]])

            query = cap_i_expand
            context = images
            for j in range(self.iteration_step):
                # "feature_update" by default:
                attn_feat, _ = func_attention(query, context, self.raw_feature_norm, smooth=self.lambda_softmax)

                row_sim = cosine_similarity(cap_i_expand, attn_feat, dim=2)
                row_sim = row_sim.mean(axis=1, keepdim=True)
                similarities[j].append(row_sim)

                query = self.gated_memory_t2i(query, attn_feat)

        # (n_image, n_caption)
        new_similarities = []
        for j in range(self.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = paddle.concat(similarities[j], 1)
            if self.training:
                similarities_one = paddle.transpose(similarities_one, (0, 1))
            new_similarities.append(similarities_one)

        return new_similarities

    def xattn_score_Image_IMRAM(self, images, captions_all, cap_lens):
        """
        Images: (batch_size, n_regions, d) matrix of images
        captions_all: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        similarities = [[] for _ in range(self.iteration_step)]
        n_image = images.shape[0]
        n_caption = captions_all.shape[0]
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0)
            cap_i_expand = paddle.expand(cap_i, [n_image, cap_i.shape[1], cap_i.shape[2]])

            query = images
            context = cap_i_expand
            for j in range(self.iteration_step):
                attn_feat, _ = func_attention(query, context, self.raw_feature_norm, smooth=self.lambda_softmax)

                row_sim = cosine_similarity(images, attn_feat, dim=2)
                row_sim = row_sim.mean(axis=1, keepdim=True)
                similarities[j].append(row_sim)

                query = self.gated_memory_i2t(query, attn_feat)

        # (n_image, n_caption)
        new_similarities = []
        for j in range(self.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = paddle.concat(similarities[j], 1)
            if self.training:
                similarities_one = paddle.transpose(similarities_one, (0, 1))
            new_similarities.append(similarities_one)

        return new_similarities

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):

        def shard_xattn_Full_IMRAM(model, images, captions, caplens, iteration_step, shard_size=128):
            """
            Computer pairwise t2i image-caption distance with locality sharding
            """
            n_im_shard = int((len(images) - 1) / shard_size) + 1
            n_cap_shard = int((len(captions) - 1) / shard_size) + 1

            # print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

            d_t2i = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]
            d_i2t = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]

            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im_emb = paddle.to_tensor(images[im_start:im_end])
                    s = paddle.to_tensor(captions[cap_start:cap_end])
                    l = paddle.to_tensor(caplens[cap_start:cap_end])
                    sim_list_t2i = model.xattn_score_Text_IMRAM(im_emb, s, l)
                    sim_list_i2t = model.xattn_score_Image_IMRAM(im_emb, s, l)
                    assert len(sim_list_t2i) == iteration_step and len(sim_list_i2t) == iteration_step
                    for k in range(iteration_step):
                        d_t2i[k][im_start:im_end, cap_start:cap_end] = sim_list_t2i[k].numpy()
                        d_i2t[k][im_start:im_end, cap_start:cap_end] = sim_list_i2t[k].numpy()

            score = 0
            for j in range(iteration_step):
                score += d_t2i[j]
            for j in range(iteration_step):
                score += d_i2t[j]

            return score

        def shard_xattn_Text_IMRAM(model, images, captions, caplens, iteration_step, shard_size=128):
            """
            Computer pairwise t2i image-caption distance with locality sharding
            """
            n_im_shard = int((len(images) - 1) / shard_size) + 1
            n_cap_shard = int((len(captions) - 1) / shard_size) + 1

            # print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

            d = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]

            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im_emb = paddle.to_tensor(images[im_start:im_end])
                    s = paddle.to_tensor(captions[cap_start:cap_end])
                    l = paddle.to_tensor(caplens[cap_start:cap_end])
                    sim_list = model.xattn_score_Text_IMRAM(im_emb, s, l)
                    assert len(sim_list) == iteration_step
                    for k in range(iteration_step):
                        d[k][im_start:im_end, cap_start:cap_end] = sim_list[k].numpy()

            score = 0
            for j in range(iteration_step):
                score += d[j]

            return score

        def shard_xattn_Image_IMRAM(model, images, captions, caplens, iteration_step, shard_size=128):
            """
            Computer pairwise t2i image-caption distance with locality sharding
            """
            n_im_shard = int((len(images) - 1) / shard_size) + 1
            n_cap_shard = int((len(captions) - 1) / shard_size) + 1

            # print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

            d = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]

            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im_emb = paddle.to_tensor(images[im_start:im_end])
                    s = paddle.to_tensor(captions[cap_start:cap_end])
                    l = paddle.to_tensor(caplens[cap_start:cap_end])
                    sim_list = model.xattn_score_Image_IMRAM(im_emb, s, l)
                    assert len(sim_list) == iteration_step
                    for k in range(iteration_step):
                        if len(sim_list[k]) != 0:
                            d[k][im_start:im_end, cap_start:cap_end] = sim_list[k].numpy()

            score = 0
            for j in range(iteration_step):
                score += d[j]
            return score

        model_mode = kwargs.get('model_mode')
        iteration_step = kwargs.get('iteration_step')
        if model_mode == 'full_IMRAM':
            sims = shard_xattn_Full_IMRAM(model, img_embs, cap_embs, cap_lens, iteration_step)
        elif model_mode == "image_IMRAM":
            sims = shard_xattn_Image_IMRAM(model, img_embs, cap_embs, cap_lens, iteration_step)
        elif model_mode == "text_IMRAM":
            sims = shard_xattn_Text_IMRAM(model, img_embs, cap_embs, cap_lens, iteration_step)
        else:
            assert False, "wrong model mode"

        return sims