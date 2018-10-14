import numpy as np
cimport numpy as cnp
from tqdm import tqdm
from collections import defaultdict
from sklearn.tree import DecisionTreeRegressor

from libcpp.vector cimport vector

ctypedef cnp.float_t DTYPE_t
ctypedef cnp.int64_t DTYPE_i

cdef extern from "../tree/ClassificationTree.h":
    cdef cppclass ClassificationTree:
        ClassificationTree(vector[vector[float]] & x, vector[int] & y, int min_samples_leaf);
        vector[int] predict(const vector[vector[float]] & x);

        const vector[vector[float]] x;
        const vector[int] y;
        int n_features;
        int nlevs;

cdef class TreeClassification:
    cdef ClassificationTree * _thisptr

    def __cinit__(self, cnp.ndarray[DTYPE_t, ndim=2] x, cnp.ndarray[DTYPE_i, ndim=1] y, int min_samples_leaf=2):
        _x = np.transpose(x)
        self._thisptr = new ClassificationTree(_x, y, min_samples_leaf)
        if self._thisptr == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    @property
    def _n_levels(self):
        return self._thisptr.nlevs

    @property
    def x(self):
        return np.transpose(self._thisptr.x)

    @property
    def y(self):
        return np.asarray(self._thisptr.y, int)

    @property
    def n_features(self):
        return self._thisptr.n_features

    def predict(self, x):
        if np.ndim(x) != 2:
            x = np.reshape(x, (1, -1))
        assert x.shape[1] == self.n_features, "x.shape[1] must match n_features."
        return np.asarray(self._thisptr.predict(x), int)

    def __call__(self, x):
        return self.predict(x)

class GBDTClassifier(object):
    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _init_estimator(self, x):
        return np.zeros(x.shape[0])

    def fit(self, x, y):
        self.n_data, self.n_features = x.shape
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self._unique_y = np.unique(y)
        self.K = len(self._unique_y)
        map_y = dict(zip(self._unique_y, np.arange(self.K)))
        self._y = np.array([map_y[iy] for iy in self.y], dtype=int)
        self.estimators_ = np.empty((self.n_estimators, self.K), dtype=np.object)
        self.values_ = np.empty((self.n_estimators, self.K), dtype=np.object)
        _F = np.asarray([self._init_estimator(self.x)] * self.K)
        for m in tqdm(range(self.n_estimators)):
            exp_Fs = np.exp(_F - np.max(_F, axis=0))
            exp_Fsum = np.sum(exp_Fs, axis=0)
            pk = exp_Fs / exp_Fsum
            for k in range(self.K):
                yk = (self._y == k) - pk[k]
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(self.x, yk)
                nodes = tree.apply(self.x)
                map_nodes = defaultdict(list)
                for i in range(self.n_data):
                    map_nodes[nodes[i]].append(i)
                values_ = dict()
                for node, indexs in map_nodes.items():
                    _yk = yk[indexs]
                    abs_yk = np.abs(_yk)
                    gamma = (self.K - 1) / self.K * np.sum(_yk) / (np.sum(abs_yk * (1 - abs_yk)) + 1e-10)
                    values_[node] = gamma
                self.estimators_[m][k] = tree
                self.values_[m][k] = values_
                _F[k] = _F[k] + np.array([values_[node] for node in nodes])

    def _predict(self, x, i):
        preds = np.zeros((self.K, x.shape[0]), dtype=float)
        for k in range(self.K):
            estimator = self.estimators_[i][k]
            values = self.values_[i][k]
            nodes = estimator.apply(x)
            preds[k] = np.array([values[node] for node in nodes], dtype=float)
        return preds

    def predict(self, x):
        if np.ndim(x) != 2:
            x = np.reshape(x, (1, -1))
        assert x.shape[1] == self.n_features, "x.shape[1] must match n_features."
        preds = np.zeros((self.K, x.shape[0]), dtype=float)
        for i in range(self.n_estimators):
            preds += self._predict(x, i)

        preds = np.argmax(preds, axis=0)
        trans_preds = np.array([self._unique_y[pred] for pred in preds])
        return trans_preds
