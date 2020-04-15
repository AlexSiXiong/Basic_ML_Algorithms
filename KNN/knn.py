import numpy as np
from math import sqrt
from collections import Counter


def KNN(k, x_train, y_train, x_predict):
    assert 1 <= k <= x_train.shape[0], "invalid k"
    assert x_train.shape[0] == y_train.shape[0] , "the size of x_train != y_train"
    # assert x_train.shape[1] == x_predict.shape[0]

    distances = [sqrt(np.sum(each_point - x_predict) ** 2) for each_point in x_train]
    nearest = np.argsort(distances)

    top_k = [y_train[i] for i in nearest[:k]]  # get the labels of the nearest points
    labels = Counter(top_k)
    result = labels.most_common(1)[0][0]

    return '预测坐标的标签是 ', result
