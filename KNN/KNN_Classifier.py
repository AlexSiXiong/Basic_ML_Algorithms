import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:
    def __init__(self, k):
        assert k >= 1, "k validation"
        self.k = k
        self._x_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        self._x_train = x_train
        self._y_train = y_train

    def predict(self, x_predict):
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self, x):
        distances = [sqrt(np.sum(each_point - x)**2) for each_point in self._x_train]
        nearest = np.argsort(distances)

        top_k = [self._y_train[i] for i in nearest[:self.k]]
        labels = Counter(top_k)
        return labels.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(K=%d)" % self.k