from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import os


class CocoReader(object):

    def __init__(self, data_root, image_root):
        """
        A class for reading COCO dataset.

        :param data_root: the folder of the dataset, the folder needs to contain the following files:
        1. dataset_coco.json, download from http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
        2. img_feat.npy and img_box.npy, processed by scrips/coco_region.py, which contain region feature an location info extracted by faster-rcnn
           origin feature can download from https://storage.googleapis.com/up-down-attention/trainval_36.zip
        3. labels.npy, processed by scrips/coco_label.py, which contains the label of images.

        :param image_root: the raw image folder of dataset, we combined train2014 and val2014.
        """

        self.train_img = []
        self.valid_img = []
        self.test_img = []
        self.train_txt = []
        self.valid_txt = []
        self.test_txt = []

        self.text = {}
        self.image = {}

        coco_data = json.load(open(os.path.join(data_root, 'dataset_coco.json'), 'r'))
        for data in coco_data['images']:
            if data['split'] == 'train' or data['split'] == 'restval':
                self.train_img.append(data['imgid'])
                self.train_txt.append(data['sentids'])
            elif data['split'] == 'val':
                self.valid_img.append(data['imgid'])
                self.valid_txt.append(data['sentids'])
            else:
                self.test_img.append(data['imgid'])
                self.test_txt.append(data['sentids'])
            self.image[data['imgid']] = data['filename']
            for sent in data['sentences']:
                self.text[sent['sentid']] = sent['raw']

        self.img_feat = np.load(os.path.join(data_root, 'img_feat.npy'), mmap_mode='r+')
        self.img_box = np.load(os.path.join(data_root, 'img_box.npy'))
        self.label = np.load(os.path.join(data_root, 'labels.npy'))

        for k, v in self.image.items():
            self.image[k] = os.path.join(image_root, v)

        # construct image to text map
        total_image = self.train_img + self.valid_img + self.test_img
        total_text = self.train_txt + self.valid_txt + self.test_txt
        self.image2text = dict(zip(total_image, total_text))

    def getImgFeat(self, idx):
        """get the region feature of image by image index"""
        return self.img_feat[idx]

    def getImgBox(self, idx):
        """get the region location of image by image index"""
        return self.img_box[idx]

    def getOneTxt(self, idx):
        """get raw text by text index"""
        return self.text[idx]

    def getAllTxt(self, idx):
        """get 5 raw text by image index cause that the ratio 1:5 for image and text"""
        ids = self.image2text[idx][:5]
        all_txt = [self.getOneTxt(i) for i in ids]
        return all_txt

    def getImg(self, idx):
        """get image file by image index"""
        return self.image[idx]

    def getLabel(self, idx):
        """get label by image index"""
        return self.label[idx]

    def train_by_image(self):
        """get train data split, one vs five"""
        return self.train_img, self.train_txt

    def valid_by_image(self):
        """get valid data split, one vs five"""
        return self.valid_img, self.valid_txt

    def test_by_image(self):
        """get test data split, one vs five"""
        return self.test_img, self.test_txt

    def train_by_text(self):
        """get train data split, one vs one"""
        cur_img = []
        cur_txt = []
        for i in range(len(self.train_img)):
            for j in range(5):
                cur_img.append(self.train_img[i])
                cur_txt.append(self.train_txt[i][j])

        return cur_img, cur_txt

    def valid_by_text(self):
        """get valid data split, one vs one"""
        cur_img = []
        cur_txt = []
        for i in range(len(self.valid_img)):
            for j in range(5):
                cur_img.append(self.valid_img[i])
                cur_txt.append(self.valid_txt[i][j])

        return cur_img, cur_txt

    def test_by_text(self):
        """get test data split, one vs one"""
        cur_img = []
        cur_txt = []
        for i in range(len(self.test_img)):
            for j in range(5):
                cur_img.append(self.test_img[i])
                cur_txt.append(self.test_txt[i][j])

        return cur_img, cur_txt

    @property
    def total_text(self):
        """return total text"""
        return list(self.text.values())

    @property
    def total_train_text(self):
        """only return train text"""
        ids = [i for t in self.train_txt for i in t]
        text = [self.text[i] for i in ids]
        return text

    @property
    def num_labels(self):
        """return nums of label, default:80"""
        return self.label.shape[1]
