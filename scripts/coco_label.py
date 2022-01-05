from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import os


def main(opt):
    # TRAIN_INS = '../data/COCO/annotations/instances_train2014.json'
    # VALID_INS = '../data/COCO/annotations/instances_val2014.json'
    # COCO_DATA = '../data/COCO/dataset_coco.json'
    # OUT_DIR = '../data/COCO/'

    TRAIN_INS = opt.train
    VALID_INS = opt.val
    COCO_DATA = opt.dataset
    OUT_DIR = opt.output

    train_data = json.load(open(TRAIN_INS, 'r'))
    val_data = json.load(open(VALID_INS, 'r'))

    data = json.load(open(COCO_DATA, 'r'))
    trans = {}
    for info in data['images']:
        trans[info['imgid']] = info['cocoid']

    cate_map = {}
    for i, cate in enumerate(train_data['categories']):
        cate_map[cate['id']] = i

    label = {}

    for info in train_data['annotations']:
        if info['image_id'] in label:
            label[info['image_id']].append(cate_map[info['category_id']])
        else:
            label[info['image_id']] = []

    for info in val_data['annotations']:
        if info['image_id'] in label:
            label[info['image_id']].append(cate_map[info['category_id']])
        else:
            label[info['image_id']] = []

    for k, v in label.items():
        label[k] = list(set(v))

    all_label = np.zeros([123287, 80], dtype=np.float32)
    for i in range(len(all_label)):
        try:
            all_label[i, label[trans[i]]] = 1.
        except:
            pass

    np.save(os.path.join(OUT_DIR, 'labels.npy'), all_label)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Path to instances_train2014.json')
    parser.add_argument('--val', type=str, help='Path to instances_val2014.json')
    parser.add_argument('--dataset', type=str, help='Path to dataset_coco.json')
    parser.add_argument('--output', type=str, help='Folder to save label file')

    opt = parser.parse_args()
    main(opt)
