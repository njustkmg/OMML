import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from torchmm.models.captioning.layers.label_smooth import LabelSmoothing


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0,
                 dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        self.norm_q = norm_q
        self.norm = LayerNorm(d_model)
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        self.do_dropout_aoa = dropout_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            self.dropout_aoa = nn.Dropout(p=dropout_aoa)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        if self.norm_q:
            query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query_, key_, value_, mask=mask,
                                 dropout=self.dropout)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if self.use_aoa:
            # Apply AoA
            if self.do_dropout_aoa:
                x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
            else:
                x = self.aoa_layer(torch.cat([x, query], -1))

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class AoA_Refiner_Layer(nn.Module):
    def __init__(self, size, self_attn, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer(x, lambda x: self.self_attn(x, x, x, mask))
        return x


class AoA_Refiner_Core(nn.Module):
    def __init__(self, num_heads, rnn_size):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(num_heads, rnn_size, project_k_v=1, scale=1,
                                       do_aoa=1, norm_q=0, dropout_aoa=0.3)
        layer = AoA_Refiner_Layer(rnn_size, attn, 0.1)
        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class AoA_Decoder_Core(nn.Module):
    def __init__(self, num_heads, rnn_size, drop_prob_lm, input_encoding_size):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = drop_prob_lm
        self.d_model = rnn_size

        self.att_lstm = nn.LSTMCell(input_encoding_size + rnn_size, rnn_size)  # we, fc, h^2_t-1
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        self.att2ctx = nn.Sequential(nn.Linear(self.d_model + rnn_size, 2 * rnn_size), nn.GLU())
        self.attention = MultiHeadedDotAttention(num_heads, rnn_size, project_k_v=0,
                                                 scale=1, use_output_layer=0, do_aoa=0, norm_q=1)

        self.ctx_drop = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        # state[0][1] is the context vector at the last step
        h_att, c_att = self.att_lstm(torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1),
                                     (state[0][0], state[1][0]))

        att = self.attention(h_att, p_att_feats.narrow(2, 0, self.d_model),
                             p_att_feats.narrow(2, self.d_model, self.d_model), att_masks)

        ctx_input = torch.cat([att, h_att], 1)
        output = self.att2ctx(ctx_input)
        # save the context vector to state[0][1]
        state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))
        output = self.out_drop(output)
        return output, state


class AoANet(nn.Module):
    def __init__(self,
                 vocab_size,
                 input_encoding_size,
                 rnn_size,
                 num_layers,
                 drop_prob_lm,
                 sample_len,
                 input_feat_size,
                 num_heads,
                 smoothing,
                 **kwargs
                 ):
        super(AoANet, self).__init__()
        self.num_layers = 2
        self.ctx2att = nn.Linear(rnn_size, 2 * rnn_size)

        self.refiner = AoA_Refiner_Core(num_heads, rnn_size)
        self.core = AoA_Decoder_Core(num_heads, rnn_size, drop_prob_lm, input_encoding_size)

        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.drop_prob_lm = drop_prob_lm
        self.seq_length = sample_len  # maximum sample length
        self.fc_feat_size = input_feat_size
        self.att_feat_size = input_feat_size

        self.use_bn = getattr(kwargs, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))

        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.criterion = LabelSmoothing(smoothing=smoothing)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def forward_xe(self, fc_feats, att_feats, seq, att_masks, cap_mask):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output

        loss = self.criterion(outputs, seq[:, 1:], cap_mask[:, 1:])

        return loss

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed att feats
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.refiner(att_feats, att_masks)

        # meaning pooling
        if att_masks is None:
            mean_feats = torch.mean(att_feats, dim=1)
        else:
            mean_feats = (torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1))

        # Project the attention feats first to reduce memory and computation.
        p_att_feats = self.ctx2att(att_feats)

        return mean_feats, att_feats, p_att_feats, att_masks

    def forward(self, batch, mode='xe', sample_method='greedy'):
        images = batch['image_feat']           # [bs, 36, 2048]
        images_loc = batch['image_loc'].float()
        images = torch.cat([images, images_loc], dim=-1)
        images_mask = batch['image_mask']      # [bs, 36]
        captions = batch['text_token']         # [bs, 5, len]
        cap_mask = batch['text_mask']          # [bs, 5, len]

        batch_size = images.shape[0]
        seq_len = captions.shape[-1]
        images = images.unsqueeze(1).expand([batch_size, 5, 36, self.att_feat_size]).reshape([batch_size*5, 36, self.att_feat_size])
        images_mask = images_mask.unsqueeze(1).expand([batch_size, 5, 36]).reshape([batch_size*5, 36])
        captions = captions.reshape([batch_size*5, seq_len])
        cap_mask = cap_mask.reshape([batch_size*5, seq_len])

        if torch.cuda.is_available():
            images = images.cuda()
            images_mask = images_mask.cuda()
            captions = captions.cuda()
            cap_mask = cap_mask.cuda()

        fc_feats = images.mean(1)
        if self.training:
            if mode == 'xe':
                return self.forward_xe(fc_feats, images, captions, images_mask, cap_mask)
            else:
                return self.forward_sample(fc_feats, images, images_mask, sample_method)
        else:
            return self.forward_sample(fc_feats, images, images_mask, sample_method)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def forward_sample(self, fc_feats, att_feats, att_masks=None, sample_method='greedy'):

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seqLogprobs, seq

    def sample_next_word(self, logprobs, sample_method):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        else:
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
        return it, sampleLogprobs
