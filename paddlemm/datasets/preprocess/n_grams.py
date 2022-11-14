from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
from collections import defaultdict


class NGrams(object):

    def __init__(self, train_text, vocab, n=4):
        """
        construct n-grams
        :param train_text: total text for extract n grams,
        note that type is [n, 5], two-dim raw text list
        :param vocab: word_to_idx dict
        :param n: default 4
        """
        self.train_text = train_text
        self.vocab = vocab
        self.n = n

    @property
    def document_frequency(self):
        ngram_words, ngram_idx = self.build_ngram()
        return ngram_idx

    def _precook(self, s):
        """Takes a string as input and returns an object that can be given to either cook_refs or cook_test.
        This is optional: cook_refs and cook_test can take string arguments as well.

        Inputs:
         - s: string : sentence to be converted into ngrams
         - n: int    : number of ngrams for which representation is calculated

         Returns:
             term frequency vector for occuring ngrams
        """
        words = s.split()
        counts = defaultdict(int)
        for k in range(1, self.n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i:i + k])
                counts[ngram] += 1
        return counts

    def _compute_doc_freq(self, crefs):
        """Compute term frequency for reference data. This will be used to compute idf
        (inverse document frequency later). The term frequency is stored in the object"""
        document_frequency = defaultdict(float)
        for refs in crefs:
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                document_frequency[ngram] += 1

        return document_frequency

    def build_ngram(self):
        refs_words = []
        refs_idx = []

        for sents in self.train_text:
            ref_words = []
            ref_idx = []
            for sent in sents:
                tokens = nltk.tokenize.word_tokenize(sent.lower()) + ['<pad>']
                tokens = [_ if _ in self.vocab else '<unk>' for _ in tokens]

                words = ' '.join(tokens)
                idx = ' '.join([str(self.vocab[_]) for _ in tokens])

                ref_words.append(self._precook(words))
                ref_idx.append(self._precook(idx))
            refs_idx.append(ref_idx)
            refs_words.append(ref_words)

        ngram_words = self._compute_doc_freq(refs_words)
        ngram_idx = self._compute_doc_freq(refs_idx)

        return ngram_words, ngram_idx
