import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np

from .layers.normalize import l2norm
from .layers.contrastive import ContrastiveLoss


# tutorials/09 - Image Captioning
class EncoderImage(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, image_norm=True):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.image_norm = image_norm
        self.use_abs = use_abs

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.cnn = models.vgg19(pretrained=True)
            for param in self.cnn.parameters():
                param.requires_grad = finetune

            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])

        elif cnn_type.startswith('resnet'):
            self.cnn = models.resnet152(pretrained=True)
            for param in self.cnn.parameters():
                param.requires_grad = finetune

            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features, 1)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if self.image_norm:
            features = l2norm(features, 1)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out, 1)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class VSEPP(nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self,
                 embed_size,
                 vocab_size,
                 word_dim,
                 num_layers,
                 finetune,
                 cnn_type,
                 margin,
                 max_violation,
                 use_abs,
                 measure,
                 image_norm=True,
                 **kwargs):
        super(VSEPP, self).__init__()
        # Build Models
        self.img_enc = EncoderImage(embed_size, finetune,
                                    cnn_type, use_abs, image_norm)
        self.txt_enc = EncoderText(vocab_size, word_dim,
                                   embed_size, num_layers,
                                   use_abs=use_abs)

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

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        lengths = lengths.tolist()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, lengths

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        cap_lens = lengths.tolist()

        # compute the embeddings
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)

        if self.mesure == 'order':
            scores = order_sim(img_emb, cap_emb)
        else:
            scores = cosine_sim(img_emb, cap_emb)

        loss = self.criterion(scores)

        return loss

