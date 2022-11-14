from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def average_precision(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the average precision
    """

    def cal_single_instance(x, y):
        idx = np.argsort(-x)
        y = y[idx]
        correct = 0
        prec = 0
        num = 0
        for i in range(x.shape[0]):
            if y[i] == 1:
                num += 1
                correct += 1
                prec += correct / (i + 1)
        return prec / num

    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    aveprec = 0
    m = 0
    for i in range(n):
        s = np.sum(y[i])
        if s in range(1, d):
            aveprec += cal_single_instance(x[i], y[i])
            m += 1
    aveprec /= m
    return aveprec


def coverage(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the test instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the coverage
    """

    def cal_single_instance(x, y):
        idx = np.argsort(x)
        y = y[idx]
        loc = x.shape[0]
        for i in range(x.shape[0]):
            if y[i] == 1:
                loc -= i
                break
        return loc

    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    cover = 0
    for i in range(n):
        cover += cal_single_instance(x[i], y[i])
    cover = cover / n - 1
    return cover


def example_auc(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the example auc
    """

    def cal_single_instance(x, y):
        idx = np.argsort(x)
        y = y[idx]
        m = 0
        n = 0
        auc = 0
        for i in range(x.shape[0]):
            if y[i] == 1:
                m += 1
                auc += n
            if y[i] == 0:
                n += 1
        auc /= (m * n)
        return auc

    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    m = 0
    auc = 0
    for i in range(n):
        s = np.sum(y[i])
        if s in range(1, d):
            auc += cal_single_instance(x[i], y[i])
            m += 1
    auc /= m
    return auc


def macro_auc(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the macro auc
    """

    def cal_single_label(x, y):
        idx = np.argsort(x)
        y = y[idx]
        m = 0
        n = 0
        auc = 0
        for i in range(x.shape[0]):
            if y[i] == 1:
                m += 1
                auc += n
            if y[i] == 0:
                n += 1
        auc /= (m * n)
        return auc

    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    auc = 0
    num = 0
    for i in range(d):
        s = np.sum(y[:, i])
        if s in range(1, n):
            num += 1
            auc += cal_single_label(x[:, i], y[:, i])
    auc /= num
    return auc


def micro_auc(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the micro auc
    """

    def cal_single_label(x, y):
        idx = np.argsort(x)
        y = y[idx]
        m = 0
        n = 0
        auc = 0
        for i in range(x.shape[0]):
            if y[i] == 1:
                m += 1
                auc += n
            if y[i] == 0:
                n += 1
        auc /= (m * n)
        return auc

    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    x = x.reshape(n * d)
    y = y.reshape(n * d)
    auc = cal_single_label(x, y)
    return auc


def ranking_loss(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the test instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise x(i,j)=0
    :return: the ranking loss
    """

    def cal_single_instance(x, y):
        idx = np.argsort(x)
        y = y[idx]
        m = 0
        n = 0
        rl = 0
        for i in range(x.shape[0]):
            if y[i] == 1:
                m += 1
            if y[i] == 0:
                rl += m
                n += 1
        rl /= (m * n)
        return rl

    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    m = 0
    rank_loss = 0
    for i in range(n):
        s = np.sum(y[i])
        if s in range(1, d):
            rank_loss += cal_single_instance(x[i], y[i])
            m += 1
    rank_loss /= m
    return rank_loss