from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def build_bow(train_text, total_text, word_dim=2912):
    """build bow feature, for cmml
    2912 follows the experimental settings in the paper **Comprehensive Semi-Supervised Multi-Modal Learning** """

    vt = CountVectorizer(max_features=word_dim)
    # only train text for bow extracting
    vt.fit(train_text)

    total_feat = vt.transform(total_text)
    total_feat = np.array(total_feat.todense(), dtype=np.float32)

    return total_feat
