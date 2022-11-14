from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import nltk
import os
import json


def build_vocab(total_text, data_root, count_thresh=5, save=True):
    """
    build vocabulary
    :param total_text: all text for build vocab
    :param data_root: path for save vocab
    :param count_thresh: The number of occurrences greater than the threshold is included in the dictionary
    :param save: whether to save
    :return: word_to_idx vocab and max length
    """
    counter = Counter()
    length = 0

    for text in total_text:
        tokens = nltk.tokenize.word_tokenize(text.lower())
        counter.update(tokens)
        length = max(length, len(tokens))

    words = [word for word, cnt in counter.items() if cnt >= count_thresh]

    vocab = {}
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<end>'] = 2
    vocab['<unk>'] = 3

    for idx, word in enumerate(words):
        vocab[word] = idx + 4

    if save:
        word2idx = vocab
        idx2word = {int(v): k for k, v in vocab.items()}
        with open(os.path.join(data_root, 'word_dict.json'), 'w') as f:
            word_dict = {
                'word2idx': word2idx,
                'idx2word': idx2word,
                'length': length
            }
            json.dump(word_dict, f)

    return vocab, length