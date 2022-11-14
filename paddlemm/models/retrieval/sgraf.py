import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from .layers.contrastive import ContrastiveLoss
from .layers.utils import l1norm, l2norm
from .layers.img_enc import EncoderImage
from .layers.txt_enc import EncoderText


class VisualSA(nn.Layer):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1D(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1D(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.weight.shape[0] + m.weight.shape[1])
                    v = np.random.uniform(-r, r, size=(m.weight.shape[0], m.weight.shape[1])).astype('float32')
                    b = np.zeros(m.bias.shape).astype('float32')
                    m.weight.set_value(v)
                    m.bias.set_value(b)
                elif isinstance(m, nn.BatchNorm1D):
                    a = np.ones(m.weight.shape).astype('float32')
                    b = np.zeros(m.bias.shape).astype('float32')
                    m.weight.set_value(a)
                    m.bias.set_value(b)

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = paddle.concat([g_emb.unsqueeze(1) for _ in range(l_emb.shape[1])], axis=1)
        common = paddle.multiply(l_emb, g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = F.softmax(weights, axis=1)

        # compute final image, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(axis=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class TextSA(nn.Layer):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.weight.shape[0] + m.weight.shape[1])
                    v = np.random.uniform(-r, r, size=(m.weight.shape[0], m.weight.shape[1])).astype('float32')
                    b = np.zeros(m.bias.shape).astype('float32')
                    m.weight.set_value(v)
                    m.bias.set_value(b)
                elif isinstance(m, nn.BatchNorm1D):
                    a = np.ones(m.weight.shape).astype('float32')
                    b = np.zeros(m.bias.shape).astype('float32')
                    m.weight.set_value(a)
                    m.bias.set_value(b)

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = paddle.concat([g_emb.unsqueeze(1) for _ in range(l_emb.shape[1])], axis=1)
        common = paddle.multiply(l_emb, g_emb)

        weights = self.embedding_common(common).squeeze(2)
        weights = F.softmax(weights, axis=1)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(axis=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class GraphReasoning(nn.Layer):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """
    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=-1)
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.weight.shape[0] + m.weight.shape[1])
                v = np.random.uniform(-r, r, size=(m.weight.shape[0], m.weight.shape[1])).astype('float32')
                b = np.zeros(m.bias.shape).astype('float32')
                m.weight.set_value(v)
                m.bias.set_value(b)
            elif isinstance(m, nn.BatchNorm1D):
                a = np.ones(m.weight.shape).astype('float32')
                b = np.zeros(m.bias.shape).astype('float32')
                m.weight.set_value(a)
                m.bias.set_value(b)

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = self.softmax(paddle.bmm(sim_query, paddle.transpose(sim_key, (0, 2, 1))))
        sim_sgr = paddle.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr


class AttentionFiltration(nn.Layer):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """
    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1D(1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.weight.shape[0] + m.weight.shape[1])
                v = np.random.uniform(-r, r, size=(m.weight.shape[0], m.weight.shape[1])).astype('float32')
                b = np.zeros(m.bias.shape).astype('float32')
                m.weight.set_value(v)
                m.bias.set_value(b)
            elif isinstance(m, nn.BatchNorm1D):
                a = np.ones(m.weight.shape).astype('float32')
                b = np.zeros(m.bias.shape).astype('float32')
                m.weight.set_value(a)
                m.bias.set_value(b)

    def forward(self, sim_emb):
        sim_attn = self.attn_sim_w(sim_emb)
        sim_attn = paddle.transpose(sim_attn, (0, 2, 1))
        sim_attn = l1norm(self.sigmoid(self.bn(sim_attn)), dim=-1)

        sim_saf = paddle.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf


class EncoderSimilarity(nn.Layer):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """
    def __init__(self, embed_size, sim_dim, module_name='AVE', sgr_step=3):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name

        self.v_global_w = VisualSA(embed_size, 0.4, 36)
        self.t_global_w = TextSA(embed_size, 0.4)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        if module_name == 'SGR':
            self.SGR_module = nn.Sequential()
            for i in range(sgr_step):
                self.SGR_module.add_sublayer(f'SGR_{i}', GraphReasoning(sim_dim))
            # self.SGR_module = nn.Sequential((GraphReasoning(sim_dim) for i in range(sgr_step))).append()
        elif module_name == 'SAF':
            self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError('Invalid input of module_name')

        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.weight.shape[0] + m.weight.shape[1])
                v = np.random.uniform(-r, r, size=(m.weight.shape[0], m.weight.shape[1])).astype('float32')
                b = np.zeros(m.bias.shape).astype('float32')
                m.weight.set_value(v)
                m.bias.set_value(b)
            elif isinstance(m, nn.BatchNorm1D):
                a = np.ones(m.weight.shape).astype('float32')
                b = np.zeros(m.bias.shape).astype('float32')
                m.weight.set_value(a)
                m.bias.set_value(b)

    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image = img_emb.shape[0]
        n_caption = cap_emb.shape[0]

        # get enhanced global images by self-attention
        img_ave = paddle.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = paddle.concat([cap_i for _ in range(n_image)], axis=0)

            # get enhanced global i-th text by self-attention
            cap_ave_i = paddle.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)

            # local-global alignment construction
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
            sim_loc = paddle.pow(paddle.subtract(Context_img, cap_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_glo = paddle.pow(paddle.subtract(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

            # concat the global and local alignments
            sim_emb = paddle.concat([sim_glo.unsqueeze(1), sim_loc], 1)

            # compute the final similarity vector
            if self.module_name == 'SGR':
                for i in range(len(self.SGR_module)):
                    sim_emb = self.SGR_module[f'SGR_{i}'](sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)

            # compute the final similarity score
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = paddle.concat(sim_all, 1)

        return sim_all


def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = paddle.transpose(query, (0, 2, 1))

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = paddle.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = paddle.transpose(attn, (0, 2, 1))
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn*smooth, axis=2)

    # --> (batch, sourceL, queryL)
    attnT = paddle.transpose(attn, (0, 2, 1))

    # --> (batch, d, sourceL)
    contextT = paddle.transpose(context, (0, 2, 1))
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = paddle.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = paddle.transpose(weightedContext, (0, 2, 1))
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext


class SGRAF(nn.Layer):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """
    def __init__(self,
                 model_name,
                 module_name,
                 sgr_step,
                 embed_size,
                 sim_dim,
                 vocab_size,
                 word_dim,
                 num_layers,
                 image_dim,
                 margin,
                 max_violation,
                 use_bi_gru=True,
                 image_norm=True,
                 text_norm=True,
                 **kwargs):

        super(SGRAF, self).__init__()

        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers,
                                   use_bi_gru=use_bi_gru, text_norm=text_norm)
        self.sim_enc = EncoderSimilarity(embed_size, sim_dim, module_name, sgr_step)

        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)

    def forward_emb(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)

        return img_embs, cap_embs, lengths

    def forward_sim(self, batch):
        img_embs, cap_embs, cap_lens = batch
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)

        return sims

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        sims = self.sim_enc(img_embs, cap_embs, lengths)
        loss = self.criterion(sims)

        return loss

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):
        shard_size = kwargs.get('shard_size', 100)

        n_im_shard = (len(img_embs) - 1) // shard_size + 1
        n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

        sims = np.zeros((len(img_embs), len(cap_embs)))
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
            for j in range(n_cap_shard):
                ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

                with paddle.no_grad():
                    im = paddle.to_tensor(img_embs[im_start:im_end], dtype='float32')
                    ca = paddle.to_tensor(cap_embs[ca_start:ca_end], dtype='float32')
                    l = paddle.to_tensor(cap_lens[ca_start:ca_end], dtype='int64')
                    sim = model.forward_sim((im, ca, l))

                sims[im_start:im_end, ca_start:ca_end] = np.array(sim)
        return sims