from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle.io import Dataset
from PIL import Image
import paddle.vision.transforms as T

from .reader import CocoReader
from .preprocess.bag_of_words import build_bow


class SemiDataset(Dataset):
    def __init__(self,
                 data_root,
                 image_root,
                 text_type,
                 image_type,
                 **kwargs):
        """
        A dataset for semi-supervised learning, especially for cmml
        **Comprehensive Semi-Supervised Multi-Modal Learning**
        """
        super(SemiDataset, self).__init__()

        self.data_root = data_root
        self.image_root = image_root
        self.image_type = image_type
        self.text_type = text_type

        self.bow_dim = kwargs.get('bow_dim', 2912)   # default 2912 from cmml paper
        self.supervised_ratio = kwargs.get('supervise_ratio', 0.3)

        # load data reader
        self.reader = CocoReader(data_root=self.data_root, image_root=self.image_root)

        # for extract global feature for image
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.train_img_id, self.train_txt_id = self.reader.train_by_text()
        self.valid_img_id, self.valid_txt_id = self.reader.valid_by_text()
        self.test_img_id, self.test_txt_id = self.reader.test_by_text()

        # build bag-of-word feature for all text
        text_dict = self.reader.text
        idx_list = list(text_dict.keys())    # total idx for text
        text_list = list(text_dict.values())      # total raw text
        total_train_list = self.reader.total_train_text     # total train text
        text_feat = build_bow(total_train_list, text_list, self.bow_dim)

        self.text = {}
        for i in range(len(idx_list)):
            self.text[idx_list[i]] = text_feat[i]

        self.split = 'train'

        # split supervised data and unsupervised data from train dataset
        permutation = np.random.permutation(len(self.train_img_id))
        self.train_img_id = np.array(self.train_img_id)
        self.train_txt_id = np.array(self.train_txt_id)

        self.train_img_id = self.train_img_id[permutation]
        self.train_txt_id = self.train_txt_id[permutation]

        self.supervised_split = int(self.supervised_ratio * 10)
        self.unsupervised_split = int((1 - self.supervised_ratio) * 10)

        self.supervised_image = self.train_img_id[:int(len(self.train_img_id) * self.supervised_ratio)]
        self.supervised_text = self.train_txt_id[:int(len(self.train_txt_id) * self.supervised_ratio)]
        self.unsupervised_image = self.train_img_id[int(len(self.train_img_id) * self.supervised_ratio):]
        self.unsupervised_text = self.train_txt_id[int(len(self.train_txt_id) * self.supervised_ratio):]

    def train_(self):
        self.split = 'train'
        return self

    def valid_(self):
        self.split = 'valid'
        return self

    def test_(self):
        self.split = 'test'
        return self

    def __getitem__(self, idx):

        if self.split == 'train':
            supervise_img = []
            supervise_text = []
            supervise_label = []
            for i in range(idx * self.supervised_split, (idx + 1) * self.supervised_split):
                temp_file = self.reader.getImg(self.supervised_image[i])
                temp_img = Image.open(temp_file).convert('RGB')
                temp_img = self.transform(temp_img)
                temp_text = self.text[self.supervised_text[i]]
                temp_label = self.reader.getLabel(self.supervised_image[i])

                temp_text = paddle.to_tensor(temp_text, dtype='float32')
                temp_label = paddle.to_tensor(temp_label, dtype='float32')
                supervise_img.append(temp_img)
                supervise_text.append(temp_text)
                supervise_label.append(temp_label)

            unsupervised_img = []
            unsupervised_text = []
            unsupervised_label = []
            for i in range(idx * self.unsupervised_split, (idx + 1) * self.unsupervised_split):
                temp_file = self.reader.getImg(self.unsupervised_image[i])
                temp_img = Image.open(temp_file).convert('RGB')
                temp_img = self.transform(temp_img)
                temp_text = self.text[self.unsupervised_text[i]]
                temp_label = self.reader.getLabel(self.unsupervised_image[i])

                temp_text = paddle.to_tensor(temp_text, dtype='float32')
                temp_label = paddle.to_tensor(temp_label, dtype='float32')
                unsupervised_img.append(temp_img)
                unsupervised_text.append(temp_text)
                unsupervised_label.append(temp_label)
            feature = []
            feature.append(supervise_text)
            feature.append(supervise_img)
            feature.append(unsupervised_text)
            feature.append(unsupervised_img)

            # return (feature, supervise_label)
            return {'feature': feature, 'label': supervise_label}

        elif self.split == 'valid':
            temp_file = self.reader.getImg(self.valid_img_id[idx])
            temp_img = Image.open(temp_file).convert('RGB')
            image = self.transform(temp_img)
            text = self.text[self.valid_txt_id[idx]]
            label = self.reader.getLabel(self.valid_img_id[idx])

            text = paddle.to_tensor(text, dtype='float32')
            label = paddle.to_tensor(label, dtype='float32')

            feature = []
            feature.append(text)
            feature.append(image)
            # return (feature, label)
            return {'feature': feature, 'label': label}

        else:
            temp_file = self.reader.getImg(self.test_img_id[idx])
            temp_img = Image.open(temp_file).convert('RGB')
            image = self.transform(temp_img)
            text = self.text[self.test_txt_id[idx]]
            label = self.reader.getLabel(self.test_img_id[idx])

            text = paddle.to_tensor(text, dtype='float32')
            label = paddle.to_tensor(label, dtype='float32')

            feature = []
            feature.append(text)
            feature.append(image)
            return {'feature': feature, 'label': label}

    def __len__(self):
        if self.split == 'train':
            return int(len(self.unsupervised_image) / self.unsupervised_split)
        elif self.split == 'valid':
            return len(self.valid_img_id)
        else:
            return len(self.test_img_id)

    @property
    def vocab_size(self):
        return None

    @property
    def num_labels(self):
        return self.reader.num_labels

    @property
    def word2idx(self):
        return None

    @property
    def idx2word(self):
        return None

    @property
    def max_len(self):
        return None