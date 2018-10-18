from abc import abstractmethod
from typing import Iterable

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np


class Model:

    @abstractmethod
    def fit(self, data, labels):
        pass

    @abstractmethod
    def predict_proba(self, test) -> Iterable[Iterable[float]]:
        pass


class RandomForest(Model):

    def __init__(self):
        self.classifier = RandomForestClassifier()

    def fit(self, data, labels):
        self.classifier.fit(data, labels)

    def predict_proba(self, test):
        return self.classifier.predict_proba(test)


class SKLearnKNN(Model):

    def __init__(self, k):
        self.k = k
        self.classifier = KNeighborsClassifier(k)

    def fit(self, data, labels):
        self.classifier.fit(data, labels)

    def predict_proba(self, test):
        return self.classifier.predict_proba(test)


class CosineKNN(Model):

    def __init__(self, k):
        self.k = k
        self.data = None
        self.labels = None
        self.get_top_k = get_top_k_func(k)

    def fit(self, data, labels):
        self.data = data
        self.labels = list(labels)

    def predict_proba(self, test):

        cosim = np.array(cosine_similarity(test, self.data))

        top_indices = np.apply_along_axis(self.get_top_k, 1, cosim)

        top_labels = np.array([[self.labels[j] for j in i[:self.k]] for i in top_indices])

        top_sim = cosim[np.arange(top_indices.shape[0])[:,None], top_indices]

        confidence_positive = np.sum(top_sim * top_labels, axis=1)
        confidence_negative = np.sum(top_sim * (1 - top_labels), axis=1)
        confidence_positive = confidence_positive / (confidence_negative + 0.0001)
        confidence_negative = 1 - confidence_positive

        return np.column_stack((confidence_negative, confidence_positive))


def get_top_k_func(k):
    k_neg = k*-1

    def get_top_k_inner(cosim_arr):
        ind = np.argpartition(cosim_arr, k_neg)[k_neg:]
        return ind[np.argsort(cosim_arr[ind])]

    return get_top_k_inner
