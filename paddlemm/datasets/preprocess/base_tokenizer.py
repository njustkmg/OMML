from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import nltk
from .build_vocab import build_vocab


class BaseTokenizer(object):

    def __init__(self,
                 total_text,
                 data_root,
                 count_thresh=5,
                 max_len=-1,
                 add_special_token=True):
        """
        A class convert raw text into numbers
        :param total_text: all text for build vocab
        :param data_root: dataset root
        :param count_thresh: The number of occurrences greater than the threshold is included in the dictionary
        :param max_len: max length of a text, manually or automatically set the longest sentence in the dataset
        :param add_special_token: whether to add <start> and <end>
        """

        self.total_text = total_text
        self.data_root = data_root
        self.count_thresh = count_thresh
        self.add_special_token = add_special_token

        if not os.path.exists(os.path.join(self.data_root, 'word_dict.json')):
            vocab, length = build_vocab(self.total_text, self.data_root, self.count_thresh)
            self.word2idx = vocab
            self.idx2word = {int(v): k for k, v in vocab.items()}
        else:
            word_dict = json.load(open(os.path.join(self.data_root, 'word_dict.json'), 'r'))
            self.word2idx = word_dict['word2idx']
            self.idx2word = word_dict['idx2word']
            self.idx2word = {int(k):v for k,v in self.idx2word.items()}
            length = word_dict['length']

        self.max_len = max_len if max_len > 0 else length + 2
        self.vocab_size = len(self.word2idx)

    def __call__(self, text):

        if not isinstance(text, list):
            text = nltk.tokenize.word_tokenize(text.lower())
            if self.add_special_token:
                num = min(len(text) + 2, self.max_len)
                mask = np.zeros(self.max_len, dtype='int64')
                mask[:num] = 1
                tokens = np.zeros(self.max_len, dtype='int64')
                tokens[0] = self.word2idx['<start>']
                tokens[num-1] = self.word2idx['<end>']

                for i, word in enumerate(text):
                    if i < self.max_len - 2:
                        if word in self.word2idx:
                            tokens[i + 1] = self.word2idx[word]
                        else:
                            tokens[i + 1] = self.word2idx['<unk>']
            else:
                num = min(len(text), self.max_len)
                mask = np.zeros(self.max_len, dtype='int64')
                mask[:num] = 1
                tokens = np.zeros(self.max_len, dtype='int64')
                tokens[0] = self.word2idx['<start>']
                tokens[num - 1] = self.word2idx['<end>']

                for i, word in enumerate(text):
                    if i < self.max_len:
                        if word in self.word2idx:
                            tokens[i] = self.word2idx[word]
                        else:
                            tokens[i] = self.word2idx['<unk>']

        else:
            # for multi sentence process
            tokens = np.zeros([5, self.max_len], dtype='int64')
            mask = np.zeros([5, self.max_len], dtype='int64')
            num = []

            if self.add_special_token:
                for idx in range(5):
                    temp = nltk.tokenize.word_tokenize(text[idx].lower())
                    n = min(len(temp) + 2, self.max_len)
                    tokens[idx][0] = self.word2idx['<start>']
                    tokens[idx][n-1] = self.word2idx['<end>']
                    mask[idx][:n] = 1
                    num.append(n)

                    for i, word in enumerate(temp):
                        if i < self.max_len - 2:
                            if word in self.word2idx:
                                tokens[idx][i + 1] = self.word2idx[word]
                            else:
                                tokens[idx][i + 1] = self.word2idx['<unk>']
            else:
                for idx in range(5):
                    temp = nltk.tokenize.word_tokenize(text[idx].lower())
                    n = min(len(temp), self.max_len)
                    mask[idx][:n] = 1
                    num.append(n)

                    for i, word in enumerate(temp):
                        if i < self.max_len:
                            if word in self.word2idx:
                                tokens[idx][i] = self.word2idx[word]
                            else:
                                tokens[idx][i] = self.word2idx['<unk>']

            num = np.array(num, dtype='int64')

        # return {'tokens': tokens, 'length': num, 'mask': mask}
        return tokens, num, mask