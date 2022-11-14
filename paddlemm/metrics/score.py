from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .caption.spice.spice import Spice
from .caption.bleu.bleu import Bleu
from .caption.cider.cider import Cider
from .caption.rouge.rouge import Rouge
from .caption.meteor.meteor import Meteor

from .caption.tokenizer.ptbtokenizer import PTBTokenizer

from .fusion import average_precision, macro_auc, micro_auc, example_auc, ranking_loss, coverage
from .retrieval import t2i, i2t

import numpy as np


def score_caption(gts, res):
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    result = {}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    for scorer, method in scorers:
        # print('computing %s score...' % (scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                result[m] = sc
                # print("%s: %0.3f" % (m, sc))
        else:
            result[method] = score
            # print("%s: %0.3f" % (method, score))

    return result


def score_fusion(gts, res):
    all_prediction = np.array(res)
    all_label = np.array(gts)

    average_precision_res = average_precision(all_prediction, all_label)
    coverage_res = coverage(all_prediction, all_label)
    example_auc_res = example_auc(all_prediction, all_label)
    macro_auc_res = macro_auc(all_prediction, all_label)
    micro_auc_res = micro_auc(all_prediction, all_label)
    ranking_loss_res = ranking_loss(all_prediction, all_label)

    result = {
        'average_precision': average_precision_res,
        'coverage': coverage_res,
        'example_auc': example_auc_res,
        'macro_auc': macro_auc_res,
        'micro_auc': micro_auc_res,
        'ranking_loss': ranking_loss_res
    }

    return result


def score_retrieval(sims, npts=1000, return_ranks=False):
    (r1, r5, r10, medr, meanr) = i2t(sims, npts, return_ranks=return_ranks)
    (r1i, r5i, r10i, medri, meanr) = t2i(sims, npts, return_ranks=return_ranks)
    recall = r1 + r5 + r10 + r1i + r5i + r10i

    result = {
        'i2t-r1': r1,
        'i2t-r5': r5,
        'i2t-r10': r10,
        'i2t-medr': medr,
        'i2t-meanr': meanr,
        't2i-r1': r1i,
        't2i-r5': r5i,
        't2i-r10': r10i,
        't2i-medr': medri,
        't2i-meanr': meanr,
        'recall': recall
    }
    return result
