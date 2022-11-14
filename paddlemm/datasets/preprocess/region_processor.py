from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class RegionProcessor(object):
    """Convert Fast-RCNN region feature to a standard text data input. """

    def __init__(self, num_boxes=36):
        # Fixed using Fast-RCNN to extract 36 features for each image
        self.num_boxes = num_boxes

    def __call__(self, img_feature, img_location):
        """
        img_feature: region feature of a image, shape [36, 2048],
        img_location: [height, weight, x1, y1, x2, y2], shape [36, 6]
        """

        img_target = np.zeros((self.num_boxes, 1601), dtype=np.float32)
        img_mask = [1] * self.num_boxes

        # img_h and img_w are same for all box.
        img_h = img_location[0][0]
        img_w = img_location[0][1]
        img_location = img_location[:, 2:]

        img_con = ((img_location[:, 3] - img_location[:, 1]) * (img_location[:, 2] - img_location[:, 0])
                   / (float(img_w) * float(img_h)))[:, np.newaxis]
        img_location[:, 0] = img_location[:, 0] / float(img_w)
        img_location[:, 1] = img_location[:, 1] / float(img_h)
        img_location[:, 2] = img_location[:, 2] / float(img_w)
        img_location[:, 3] = img_location[:, 3] / float(img_h)
        img_location = np.concatenate([img_location, img_con], axis=1)

        # return {'region': np.array(img_feature),
        #         'location': np.array(img_location),
        #         'mask': np.array(img_mask),
        #         'target': np.array(img_target)}
        return np.array(img_feature, dtype=np.float32), \
               np.array(img_location, dtype=np.float32), \
               np.array(img_mask), \
               np.array(img_target)