from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import numpy as np

from .basic_dataset import BasicDataset


class PretrainDataset(BasicDataset):

    def __init__(self,
                 data_root,
                 image_root,
                 text_type,
                 image_type,
                 **kwargs):
        """
        For ViLBERT pretrained
        Two task:
            1. masked multi-modal modelling
            2. multi-modal alignment prediction
        Args:
        :param text_type: token, bert
        :param image_type: raw, region
        """
        super().__init__(data_root,
                         image_root,
                         text_type,
                         image_type,
                         **kwargs)

    def __getitem__(self, idx):

        caption = self.reader.getOneTxt(self.txt_id[idx])
        image_feat = self.reader.getImgFeat(self.img_id[idx])
        image_loc = self.reader.getImgBox(self.img_id[idx])

        caption, is_next = self._random_caption(caption)
        overlaps = self._iou(image_loc[:, 2:], image_loc[:, 2:])

        text_token, text_len, text_mask = self.bert_tokenizer(caption)
        image_feat, image_loc, image_mask, image_target = self.region_processor(image_feat, image_loc)

        text_segment = np.zeros_like(text_mask)
        text_token, text_label = self._random_word(text_token)
        image_feat, image_label, masked_label = self._random_region(image_feat, overlaps)

        # add global token for image
        sum_count = max(1, int(np.sum(masked_label == 0)))
        g_image_feat = np.sum(image_feat, axis=0) / sum_count
        image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=0), image_feat], axis=0)
        image_feat = np.array(image_feat, dtype=np.float32)
        g_image_loc = np.array([0, 0, 1, 1, 1], dtype=np.float32)
        image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=0), image_loc], axis=0)
        image_loc = np.array(image_loc, dtype=np.float32)
        g_image_mask = np.array([1])
        image_mask = np.concatenate([g_image_mask, image_mask], axis=0)

        return {
            'text_token': torch.LongTensor(text_token),
            'text_mask': torch.LongTensor(text_mask),
            'text_segment': torch.LongTensor(text_segment),
            'text_label': torch.LongTensor(text_label),
            'is_next': is_next,
            'image_feat': image_feat,
            'image_loc': image_loc,
            'image_target': image_target,
            'image_label': image_label,
            'image_mask': image_mask
        }

    def __len__(self):
        return len(self.img_id)

    def _random_caption(self, caption):
        """Randomly scramble text and pictures"""
        if random.random() > 0.5:
            rand_idx = random.randint(0, len(self.txt_id))
            caption = self.reader.getOneTxt(self.txt_id[rand_idx])
            is_next = 1
        else:
            is_next = 0

        return caption, is_next

    def _random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            if i == 0 or i == len(tokens) - 1:
                output_label.append(-1)
                continue
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.bert_tokenizer.mask_token
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(self.bert_tokenizer.vocab_size)
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def _random_region(self, img_feature, overlaps):
        output_label = []
        masked_label = np.zeros((img_feature.shape[0]))

        for i in range(len(img_feature)):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.9:
                    img_feature[i] = 0
                # mask the overlap regions into zeros
                masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)
                # 10% randomly change token to random token
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        output_label = np.array(output_label)
        return img_feature, output_label, masked_label

    def _iou(self, anchors, gt_boxes):
        """
        anchors: (N, 4) ndarray of float
        gt_boxes: (K, 4) ndarray of float
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        N = anchors.shape[0]
        K = gt_boxes.shape[0]

        gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)).reshape(1, K)
        anchors_area = ((anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)).reshape(N, 1)

        boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
        query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

        iw = (np.minimum(boxes[:, :, 2], query_boxes[:, :, 2]) - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0]) + 1)
        iw[iw < 0] = 0
        ih = (np.minimum(boxes[:, :, 3], query_boxes[:, :, 3]) - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1]) + 1)
        ih[ih < 0] = 0

        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        return overlaps
