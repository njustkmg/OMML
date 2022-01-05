import copy
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .layers.label_smooth import LabelSmoothing


def clones(module, N):
    "Produce N identical layers."
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention.
    Inputs:
     - query: [batch_size, num_heads, num_objects, feat_dim]
     - key: [batch_size, num_heads, num_objects, feat_dim]
     - value: [batch_size, num_heads, num_objects, feat_dim]
     - mask: [batch_size, 1, 1, num_objects]
     - dropout: dropout prob.
    """
    d_k = query.shape[-1]
    # [batch_size, num_heads, num_objects, num_objects]
    scores = paddle.matmul(query, key.transpose([0, 1, 3, 2])) / math.sqrt(d_k)

    if mask is not None:
        scores_mask = paddle.fluid.layers.fill_constant(shape=scores.shape, dtype=scores.dtype, value=-1e9)
        scores = paddle.where(paddle.broadcast_to(mask, shape=scores.shape) != 0, scores, scores_mask)
    p_attn = F.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return paddle.matmul(p_attn, value), p_attn


class GLU(nn.Layer):
    """Applies the gated linear unit function."""

    def __init__(self, dim=-1):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, input):
        return F.glu(input, axis=self.dim)


class SublayerConnection(nn.Layer):
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


class LayerNorm(nn.Layer):
    """Construct a layernorm module. See equation (7) in the paper for details."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = paddle.create_parameter(shape=(features,), dtype='float32',
                                           default_initializer=nn.initializer.Constant(value=1.))
        self.b_2 = paddle.create_parameter(shape=(features,), dtype='float32',
                                           default_initializer=nn.initializer.Constant(value=0.))
        self.eps = eps

    def forward(self, x):
        mean = paddle.mean(x, axis=-1, keepdim=True)
        std = paddle.std(x, axis=-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadAttention(nn.Layer):
    def __init__(self, h, d_model, dropout=0.1, project_k_v=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        """Inputs:
         - h: number of heads, default is 8.
         - d_model: dim of feature.
         - dropout: dropout prob for attention
         - project_k_v: do we need to do linear projections on K and V?
         - do_aoa: whether utilize aoa to refine the feature
         - norm_q: whether to norm the query
         - dropout_aoa: dropout prob of aoa
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume the dims of K and V are equal
        self.d_k = d_model // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x

        self.linears = clones(module=nn.Linear(d_model, d_model), N=1 + 2 * project_k_v)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        """Inputs:
         - query: [batch_size, num_objects, rnn_size]
         - value: [batch_size, num_objects, rnn_size]
         - key: [batch_size, num_objects, rnn_size]
         - masks: [batch_size, num_objects]
        """
        if mask is not None:
            if len(mask.shape) == 2:
                mask = paddle.unsqueeze(mask, axis=-2)
            # same mask applied to all h heads
            mask = paddle.unsqueeze(mask, axis=1)

        single_query = 0
        if len(query.shape) == 2:
            single_query = 1
            query = paddle.unsqueeze(query, axis=1)

        n_batch = query.shape[0]
        query = self.norm(query)

        # do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
            key_ = key.reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
            value_ = value.reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
        else:
            query_, key_, value_ = \
                [l(x).reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
                 for l, x in zip(self.linears, (query, key, value))]
        # apply attention on all the projected vectors in batch
        # x: [batch_size, num_heads, num_objects, rnn_size]
        # self.attn: [batch_size, num_heads, num_objects, num_objects]
        # see equation (8), (9), (10) in the paper for details
        x, self.attn = attention(query_, key_, value_, mask=mask, dropout=self.dropout)
        # concat
        # [batch_size, num_objects, rnn_size]
        x = x.transpose([0, 2, 1, 3]).reshape((n_batch, -1, self.h * self.d_k))

        if self.use_aoa:
            # apply AoA
            # see equation (6) for details
            x = self.aoa_layer(self.dropout_aoa(paddle.concat([x, query], axis=-1)))

        if single_query:
            query = paddle.squeeze(query, axis=1)
            x = paddle.squeeze(x, axis=1)
        return x


class AoA_Refiner_Layer(nn.Layer):
    def __init__(self, size, self_attn, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer(x, lambda x: self.self_attn(x, x, x, mask))
        return x


class AoA_Refiner_Core(nn.Layer):
    def __init__(self,
                 num_heads,
                 rnn_size):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadAttention(num_heads, rnn_size,
                                  project_k_v=1,
                                  do_aoa=1, norm_q=0,
                                  dropout_aoa=0.3)
        layer = AoA_Refiner_Layer(rnn_size, attn, 0.1)
        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Inputs:
         - embeded_att_feats: [batch_size, num_objects, rnn_size]
         - mask: [batch_size, num_objects]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class AoA_Decoder_Core(nn.Layer):
    def __init__(self,
                 num_heads,
                 drop_prob_lm,
                 rnn_size,
                 input_encoding_size):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = drop_prob_lm
        self.d_model = rnn_size
        self.att_lstm = nn.LSTMCell(input_encoding_size + rnn_size, rnn_size)
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        # AoA layer
        self.att2ctx = nn.Sequential(
            nn.Linear(self.d_model + rnn_size, 2 * rnn_size), GLU())

        self.attention = MultiHeadAttention(num_heads, rnn_size, project_k_v=0, do_aoa=0, norm_q=1)

        self.ctx_drop = nn.Dropout(self.drop_prob_lm)

    def forward(self, word_emb, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        """Inputs:
         - word_emb: [batch_size, input_encoding_size]
         - mean_feats: [batch_size, rnn_size]
         - att_feats: [batch_size, num_objects, rnn_size]
         - p_att_feats: [batch_size, num_objects, rnn_size * 2]
         - state: hidden state and memory cell of lstm.
         - att_mask: [batch_size, num_objects]
        """
        # state[0][1] is the context vector at the last step
        prev_h = state[0][1]

        # the input vector to the attention lstm consists of the previous output of lstm (prev_h),
        # mean-pooled image feature (mean_feats) and an encoding of the previous generated word (word_embs).
        # see equation (12) in the paper for details.
        att_lstm_input = paddle.concat([word_emb, mean_feats + self.ctx_drop(prev_h)], axis=1)
        _, (h_att, c_att) = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att,
                             paddle.slice(p_att_feats, axes=[2], starts=[0], ends=[self.d_model]),
                             paddle.slice(p_att_feats, axes=[2], starts=[self.d_model], ends=[self.d_model * 2]),
                             att_masks)
        ctx_input = paddle.concat([att, h_att], axis=1)
        output = self.att2ctx(ctx_input)

        # save the context vector to state[0][1]
        state = (paddle.concat([paddle.unsqueeze(h_att, 0), paddle.unsqueeze(output, 0)]),
                 paddle.concat([paddle.unsqueeze(c_att, 0), paddle.unsqueeze(state[1][1], 0)]))
        output = self.out_drop(output)
        return output, state


class AoANet(nn.Layer):
    """Implementation of the attention on attention model."""

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
                 **kwargs):
        super(AoANet, self).__init__()
        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.drop_prob_lm = drop_prob_lm
        self.seq_length = sample_len  # maximum sample length
        self.fc_feat_size = input_feat_size
        self.att_feat_size = input_feat_size
        self.num_heads = num_heads
        # self.att_hid_size = att_hid_size
        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))

        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(self.drop_prob_lm))

        # self.classifier = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.classifier = nn.Linear(self.rnn_size, self.vocab_size)
        self.ctx2att = nn.Linear(self.rnn_size, 2 * self.rnn_size)

        self.refiner = AoA_Refiner_Core(self.num_heads, self.rnn_size)
        self.core = AoA_Decoder_Core(self.num_heads, self.drop_prob_lm, self.rnn_size, self.input_encoding_size)

        self.criterion = LabelSmoothing(vocab_size=vocab_size, smoothing=smoothing)
        # self.criterion_m = MaskCrossEntropy()
        # self.criterion_r = RewardLoss()

    def _clip_att(self, att_feats, att_masks):
        """Clip the length of att_masks and att_feats to the maximum length

        Inputs:
         - att_feats: [batch_size, num_objects, att_dim]
         - att_masks: [batch_size, num_objects]
        """
        if att_masks is not None:
            max_len = paddle.cast(att_masks, dtype='int64').sum(axis=1).max()
            att_feats = att_feats[:, :max_len]
            att_masks = att_masks[:, :max_len]
        return att_feats, att_masks

    def _init_hidden(self, batch_size):
        """Init hidden state and cell memory for lstm."""
        return (paddle.zeros([self.num_layers, batch_size, self.rnn_size]),
                paddle.zeros([self.num_layers, batch_size, self.rnn_size]))

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        """Embed att_feats, and prepare att_feats for computing attention later.
        Inputs:
         - fc_feats: [batch_size, fc_feat_dim]
         - att_feats: [batch_size, num_objects, att_feat_dim]
         - att_masks: [batch_size, num_objects]
        """
        # embed att feats
        att_feats = self.att_embed(att_feats)
        scores_mask = paddle.full(shape=att_masks.shape, dtype=att_masks.dtype, fill_value=1e-9)
        scores = paddle.where(paddle.broadcast_to(att_masks, shape=scores_mask.shape) != 0, att_masks, scores_mask)
        att_feats *= paddle.unsqueeze(scores, axis=[-1])
        att_feats = self.refiner(att_feats, att_masks)

        # meaning pooling
        # use mean_feats instead of fc_feats
        if att_masks is None:
            mean_feats = paddle.mean(att_feats, axis=1)
        else:
            mean_feats = (paddle.sum(att_feats * paddle.unsqueeze(att_masks, axis=[-1]), axis=1) /
                          paddle.sum(paddle.unsqueeze(att_masks, axis=[-1]), axis=1))

        # Project the attention feats first to reduce memory and computation.
        p_att_feats = self.ctx2att(att_feats)

        return mean_feats, att_feats, p_att_feats, att_masks

    def _forward_step(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        """Forward the LSTM each time step.

        Inputs:
         - it: previous generated words. paddle.LongTensor of shape [batch_size, ].
         - fc_feats: paddle.FloatTensor of shape [batch_size, rnn_size].
         - att_feats: paddle.FloatTensor of shape [batch_size, num_objects, rnn_size].
         - pre_att_feats: paddle.FloatTensor of shape [batch_size, num_objects, rnn_size * 2]
         - state: hidden state and memory cell of lstm.
        """
        word_embs = self.embed(it)
        output, state = self.core(word_embs, fc_feats, att_feats, p_att_feats, state, att_masks)
        output = self.classifier(output)

        logprobs = nn.functional.log_softmax(output, axis=1)
        return output, logprobs, state

    def forward(self, batch, mode='xe', sample_method='greedy'):
        images = batch['image_feat']           # [bs, 36, 2048]
        images_mask = batch['image_mask']      # [bs, 36]
        captions = batch['text_token']         # [bs, 5, len]
        cap_mask = batch['text_mask']          # [bs, 5, len]

        batch_size = images.shape[0]
        seq_len = captions.shape[-1]
        images = images.unsqueeze(1).expand([batch_size, 5, 36, 2048]).reshape([batch_size*5, 36, 2048])
        images_mask = images_mask.unsqueeze(1).expand([batch_size, 5, 36]).reshape([batch_size*5, 36])
        captions = captions.reshape([batch_size*5, seq_len])
        cap_mask = cap_mask.reshape([batch_size*5, seq_len])

        fc_feats = images.mean(1)
        if self.training:
            if mode == 'xe':
                return self.forward_xe(fc_feats, images, captions, images_mask, cap_mask)
            else:
                return self.forward_sample(fc_feats, images, images_mask, sample_method)
        else:
            return self.forward_sample(fc_feats, images, images_mask, sample_method)

    def forward_xe(self, fc_feats, att_feats, seq, att_masks, seq_mask):
        """Train the captioner with cross-entropy loss.

         - fc_feats: [batch_size, fc_feat_dim]
         - att_feats: [batch_size, num_objects, att_feat_dim]
         - seq: [batch_size, seq_len]
         - att_masks: [batch_size, num_objects]
        """
        batch_size = fc_feats.shape[0]

        # prepare feats
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        # init lstm state
        state = self._init_hidden(batch_size)

        logit_outputs = []
        prob_outputs = []
        # this is because we add start and end token into the caption
        for i in range(seq.shape[1] - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                # using scheduled sampling
                sample_prob = paddle.uniform(shape=(batch_size,), min=0, max=1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()  # [batch_size, ]
                else:
                    sample_ind = sample_mask.nonzero().reshape((-1,))
                    it = seq[:, i].clone()  # # [batch_size, ]
                    prob_prev = prob_outputs[i - 1].detach().exp()

                    index_selected = paddle.index_select(x=paddle.multinomial(prob_prev, num_samples=1).reshape((-1,)),
                                                         index=sample_ind, axis=0)

                    assert index_selected.shape[0] == sample_ind.shape[0]
                    # replace the groundtruth word with generated word when sampling next word
                    for j, ind in enumerate(sample_ind):
                        it[ind] = index_selected[j]
            else:
                it = seq[:, i].clone()

            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, logprobs, state = self._forward_step(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # used for compute loss
            logit_outputs.append(output)
            # used for sample words
            prob_outputs.append(logprobs)

        # we concat the output when finish all time steps
        logit_outputs = paddle.stack(logit_outputs, axis=1)
        prob_outputs = paddle.stack(prob_outputs, axis=1)  # [batch_size, max_len, vocab_size]

        loss = self.criterion(prob_outputs, seq[:, 1:], seq_mask[:, 1:])
        return loss
        # return logit_outputs, prob_outputs

    def forward_sample(self, fc_feats, att_feats, att_masks=None, sample_method='greedy'):

        batch_size = fc_feats.shape[0]
        state = self._init_hidden(batch_size)

        # prepare feats
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        # seq = paddle.zeros((batch_size, self.seq_length), dtype='int64')
        # seqLogprobs = paddle.zeros((batch_size, self.seq_length))
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length):
            if t == 0:  # input <bos>
                it = paddle.zeros((batch_size,), dtype='int64')

            _, logprobs, state = self._forward_step(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break

            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished.cast('float32') * (it > 0).cast('float32')

            it = it * unfinished.cast(it.dtype)
            # seq[:, t] = it
            seq.append(it)
            # seqLogprobs[:, t] = sampleLogprobs.reshape((-1, ))
            seqLogprobs.append(sampleLogprobs.reshape((-1,)))

            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        # we concat the output when finish all time steps
        seq = paddle.stack(seq, axis=1).cast('int64')
        seqLogprobs = paddle.stack(seqLogprobs, axis=1)

        return seqLogprobs, seq

    def sample_next_word(self, logprobs, sample_method):
        if sample_method == 'greedy':
            sampleLogprobs = paddle.max(logprobs, 1)
            it = paddle.argmax(logprobs, 1, True)
            it = it.reshape((-1,)).cast('int64')
        elif sample_method == 'sample':
            probs = paddle.exp(logprobs)
            it = paddle.multinomial(probs, 1)
            it = it.reshape((-1,)).cast('int64')
            # prepare data for paddle.gather_nd
            batch_size = it.shape[0]
            gather_index = paddle.zeros((batch_size, 2), dtype='int64')  # [batch_size, 2]
            gather_index[:, 0] = paddle.arange(batch_size)
            gather_index[:, 1] = it
            # gather the logprobs at sampled positions
            sampleLogprobs = paddle.gather_nd(logprobs, gather_index)
        else:
            raise ValueError('No such sample method')

        return it, sampleLogprobs