import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from .utils import l2norm


def EncoderImage(model_name, img_dim, embed_size, image_norm=True, cnn_type=None, finetune=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    model_name = model_name.lower()
    EncoderMap = {
        'scan': EncoderImagePrecomp,
        'vsepp': EncoderImageFull,
        'sgraf': EncoderImagePrecomp,
        'imram': EncoderImagePrecomp,
        'bfan': EncoderImagePrecomp,
    }

    if model_name in EncoderMap:
        img_enc = EncoderMap[model_name](img_dim, embed_size, image_norm, cnn_type, finetune)
    else:
        raise ValueError("Unknown model: {}".format(model_name))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, image_norm=True, cnn_type=None, finetune=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.image_norm = image_norm
        self.fc = nn.Linear(img_dim, embed_size)

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
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if self.image_norm:
            features = l2norm(features, dim=-1)

        return features


class EncoderImageFull(nn.Module):

    def __init__(self, img_dim, embed_size, image_norm=True, cnn_type='vgg19', finetune=False):

        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.image_norm = image_norm

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.cnn = models.vgg19(pretrained=True)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])

            for param in self.cnn.parameters():
                param.requires_grad = finetune

            self.fc = nn.Linear(4096, 1024)

        elif cnn_type.startswith('resnet'):
            self.cnn = models.resnet152(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            for param in self.cnn.parameters():
                param.requires_grad = finetune

            self.fc = nn.Linear(2048, 1024)

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
        features = self.cnn(images).squeeze()

        # normalization in the image embedding space
        features = l2norm(features, 1)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if self.image_norm:
            features = l2norm(features, 1)

        return features
