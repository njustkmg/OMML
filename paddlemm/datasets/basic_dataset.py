from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle.io import Dataset
from PIL import Image
import paddle.vision.transforms as T

from .reader import CocoReader
from .preprocess.base_tokenizer import BaseTokenizer
from .preprocess.region_processor import RegionProcessor
from .preprocess.pre_tokenizer import PreTokenizer


class BasicDataset(Dataset):

    def __init__(self,
                 data_root,
                 image_root,
                 text_type,
                 image_type,
                 **kwargs):
        """
        Basic processing of image and text
            For image: Provide original picture and regional feature loading
            For text: Provide text processing into token
            Label info (Optional)
        :param text_type: token, bert
        :param image_type: raw, region
        """
        super(BasicDataset, self).__init__()

        self.data_root = data_root
        self.image_root = image_root
        self.text_type = text_type
        self.image_type = image_type

        self.count_thresh = kwargs.get('count_thresh', 5)
        self.token_len = kwargs.get('max_len', -1)
        self.num_boxes = kwargs.get('num_boxes', 36)
        self.add_special_token = kwargs.get('add_special_token', True)

        # use one vs one or one vs five between image and text
        if kwargs.get('model_name').lower() == 'aoanet':
            self.ratio = 5
        else:
            self.ratio = 1

        # load data reader
        self.reader = CocoReader(data_root=self.data_root, image_root=self.image_root)

        # default is train dataset
        if self.ratio == 1:
            self.img_id, self.txt_id = self.reader.train_by_text()
        else:
            self.img_id, self.txt_id = self.reader.train_by_image()

        # load data preprocess
        self.tokenizer = BaseTokenizer(total_text=self.reader.total_text,
                                       data_root=self.data_root,
                                       count_thresh=self.count_thresh,
                                       max_len=self.token_len,
                                       add_special_token=self.add_special_token)

        if text_type == 'bert':
            self.bert_tokenizer = PreTokenizer(max_len=kwargs.get('max_len', 32),
                                               bert_model=kwargs.get('bert_model', 'bert-base-uncased'),
                                               add_special_token=self.add_special_token)

        # for extract global feature for image
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.region_processor = RegionProcessor(num_boxes=self.num_boxes)

    def train_(self):
        if self.ratio == 1:
            self.img_id, self.txt_id = self.reader.train_by_text()
        else:
            self.img_id, self.txt_id = self.reader.train_by_image()

        return self

    def valid_(self):
        if self.ratio == 1:
            self.img_id, self.txt_id = self.reader.valid_by_text()
        else:
            self.img_id, self.txt_id = self.reader.valid_by_image()

        return self

    def test_(self):
        if self.ratio == 1:
            self.img_id, self.txt_id = self.reader.test_by_text()
        else:
            self.img_id, self.txt_id = self.reader.test_by_image()

        return self

    def __getitem__(self, idx):

        if self.image_type == 'raw':
            # use global feature
            img_file = self.reader.getImg(self.img_id[idx])
            image = Image.open(img_file).convert('RGB')
            image_feat = self.transform(image)
            image_loc = 'None'
            image_mask = 'None'
            image_target = 'None'
        elif self.image_type == 'region':
            # use region feature
            feat = self.reader.getImgFeat(self.img_id[idx])
            box = self.reader.getImgBox(self.img_id[idx])
            image_feat, image_loc, image_mask, image_target = self.region_processor(feat, box)
        else:
            raise ValueError('No such image processing method!')

        if self.text_type == 'token':
            if self.ratio == 1:
                txt = self.reader.getOneTxt(self.txt_id[idx])
            else:
                txt = self.reader.getAllTxt(self.img_id[idx])
            text_token, text_len, text_mask = self.tokenizer(txt)
        else:
            raise ValueError('No such text processing method!')

        # for classification
        label = self.reader.getLabel(self.img_id[idx])

        # all text for a image, for image caption test
        all_txt = self.reader.getAllTxt(self.img_id[idx])

        return {
            'image_feat': paddle.to_tensor(image_feat, dtype='float32'),
            'image_loc': image_loc,
            'image_mask': image_mask,
            'image_target': image_target,
            'text_token': paddle.to_tensor(text_token, dtype='int64'),
            'text_len': paddle.to_tensor(text_len, dtype='int64'),
            'text_mask': paddle.to_tensor(text_mask, dtype='int64'),
            'label': paddle.to_tensor(label, dtype='float32'),
            'all_text': all_txt
        }

    def __len__(self):
        return len(self.img_id)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def num_labels(self):
        return self.reader.num_labels

    @property
    def word2idx(self):
        return self.tokenizer.word2idx

    @property
    def idx2word(self):
        return self.tokenizer.idx2word

    @property
    def max_len(self):
        return self.tokenizer.max_len