from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import paddle
import numpy as np

from .basic_dataset import BasicDataset


class SampleDataset(BasicDataset):

    def __init__(self,
                 data_root,
                 image_root,
                 text_type,
                 image_type,
                 **kwargs):
        """
        Provide random loading of negative samples based on BasicDataset
        :param text_type: token, bert
        :param image_type: raw, region
        """
        super().__init__(data_root,
                         image_root,
                         text_type,
                         image_type,
                         **kwargs)

    def __getitem__(self, idx):

        caption_pos = self.reader.getOneTxt(self.txt_id[idx])
        feat_pos = self.reader.getImgFeat(self.img_id[idx])
        box_pos = self.reader.getImgBox(self.img_id[idx])
        image_feat_pos, image_loc_pos, image_mask_pos, image_target_pos = self.region_processor(feat_pos, box_pos)

        # negative samples
        # 1: correct one, 2: random caption wrong, 3: random image wrong.
        caption_neg = self._sample_neg_caption(idx)
        feat_neg, box_neg = self._sample_neg_img(idx)
        image_feat_neg, image_loc_neg, image_mask_neg, image_target_neg = self.region_processor(feat_neg, box_neg)

        caption = [caption_pos, caption_neg, caption_pos]
        text_token, text_len, text_mask = self.bert_tokenizer(caption)

        image_feat = np.stack([image_feat_pos, image_feat_pos, image_feat_neg])
        image_loc = np.stack([image_loc_pos, image_loc_pos, image_loc_neg])
        image_mask = np.stack([image_mask_pos, image_mask_pos, image_mask_neg])
        image_target = np.stack([image_target_pos, image_target_pos, image_target_neg])

        return {
            'image_feat': image_feat,
            'image_loc': image_loc,
            'image_mask': image_mask,
            'image_target': image_target,
            'text_token': paddle.to_tensor(text_token, dtype='int64'),
            'text_len': text_len,
            'text_mask': paddle.to_tensor(text_mask, dtype='int64'),
            'label': 'None',
            'all_text': 'None'
        }

    def __len__(self):
        return len(self.img_id)

    def _sample_neg_caption(self, idx):
        while True:
            rand_idx = random.randint(0, len(self.txt_id))
            if rand_idx != idx:
                break
        caption_idx = self.txt_id[rand_idx]
        caption = self.reader.getOneTxt(caption_idx)
        return caption

    def _sample_neg_img(self, idx):
        idx = idx
        while True:
            rand_idx = random.randint(0, len(self.img_id))
            if rand_idx != idx:
                break
        feat = self.reader.getImgFeat(self.img_id[rand_idx])
        box = self.reader.getImgBox(self.img_id[idx])
        return feat, box
