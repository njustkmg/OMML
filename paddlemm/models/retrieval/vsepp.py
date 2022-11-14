import paddle.nn as nn

from .layers.txt_enc import EncoderText
from .layers.img_enc import EncoderImage
from .layers.utils import cosine_sim, order_sim
from .layers.contrastive import ContrastiveLoss


class VSEPP(nn.Layer):

    def __init__(self,
                 model_name,
                 embed_size,
                 vocab_size,
                 word_dim,
                 num_layers,
                 finetune,
                 cnn_type,
                 margin,
                 max_violation,
                 measure,
                 use_bi_gru=False,
                 image_norm=True,
                 text_norm=False,
                 **kwargs):
        super(VSEPP, self).__init__()
        # Build Models
        image_dim = None
        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm,
                                    cnn_type=cnn_type, finetune=finetune)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers,
                                   use_bi_gru=use_bi_gru, text_norm=text_norm)

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=margin,
                                         max_violation=max_violation)
        self.mesure = measure

    def forward_emb(self, batch):
        """Compute the image and caption embeddings
        """
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, lengths

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        # compute the embeddings
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)

        if self.mesure == 'order':
            scores = order_sim(img_emb, cap_emb)
        else:
            scores = cosine_sim(img_emb, cap_emb)

        loss = self.criterion(scores)

        return loss

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):
        sims = img_embs.dot(cap_embs.T)
        return sims