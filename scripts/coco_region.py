"""
http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
https://storage.googleapis.com/up-down-attention/trainval_36.zip
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import base64
import sys
import os
import numpy as np
import json

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def main(opt):
    INPUT_DATA = '../../COCO/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
    COCO_DATA = '../../COCO/caption_datasets/dataset_coco.json'
    OUT_DIR = '../../COCO/'

    # INPUT_DATA = opt.rcnn
    # COCO_DATA = opt.dataset
    # OUT_DIR = opt.output

    data = json.load(open(COCO_DATA, 'r'))
    trans = {}
    for info in data['images']:
        trans[info['imgid']] = info['cocoid']

    feature = {}
    location = {}
    for v in trans.values():
        feature[v] = None
        location[v] = None

    with open(INPUT_DATA, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            if item['image_id'] in feature:
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.decodestring(bytes(item[field], encoding="utf8")),
                                                dtype=np.float32).reshape((int(item['num_boxes']), -1))
                feature[item['image_id']] = item['features']

                # location: h, w, x1, y1, x2, y2
                temp_location = []
                image_h = int(item['image_h'])
                image_w = int(item['image_w'])

                for box in item['boxes']:
                    temp_location.append([image_h, image_w] + box.tolist())
                location[item['image_id']] = temp_location

    data_out = np.stack([feature[trans[i]] for i in range(123287)], axis=0)
    print("Final feature numpy array shape:", data_out.shape)
    np.save(os.path.join(OUT_DIR, 'img_feat.npy'), data_out)

    box_out = np.stack([location[trans[i]] for i in range(123287)], axis=0)
    print("Final box numpy array shape:", box_out.shape)
    np.save(os.path.join(OUT_DIR, 'img_box.npy'), box_out)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rcnn', type=str, help='Path to trainval_resnet101_faster_rcnn_genome_36.tsv')
    parser.add_argument('--dataset', type=str, help='Path to dataset_coco.json')
    parser.add_argument('--output', type=str, help='Folder to save region feature and box info')

    opt = parser.parse_args()
    main(opt)