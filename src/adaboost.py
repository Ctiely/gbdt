import numpy as np
from tqdm import tqdm
from GBDT import TreeClassification


#def sampling(x, size, p):
#    x = np.array(x)
#    p = np.array(p)
#    n = len(x)
#    assert n == len(p)
#    p_copy = p.copy()
#    for i in range(1, n):
#        p_copy[i] += p_copy[i - 1]
#    p_copy = p_copy / p_copy[-1]
#    samples = np.empty((size))
#
#    for s in range(size):
#        u = np.random.random()
#        for i in range(n):
#            if p_copy[i] > u:
#                samples[s] = x[i]
#                break
#    return samples

class AdaBoostClassification(object):
    def __init__(self, n_estimators=100, min_samples_leaf=10, max_depth=3):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        
    def _init_weight(self):
        return np.ones(self.n_samples, dtype=np.double)

    def fit(self, x, y):
        self.x = np.array(x)
        assert np.ndim(x) == 2
        self.y = np.array(y, copy=True).reshape(-1)
        unique_y = np.unique(self.y)
        self.K = len(unique_y)

        map_y = dict(zip(unique_y, np.arange(self.K)))
        self._y = np.array([map_y[iy] for iy in self.y], dtype=int)
        self.n_samples, self.n_features = x.shape

        self.estimators_ = []
        self.values_ = []
        sample_weight = self._init_weight()
        for m in tqdm(range(self.n_estimators), ncols=min(100, self.n_estimators)):
            estimator = TreeClassification(self.x,
                                           self._y,
                                           sample_weight,
                                           min_samples_leaf=self.min_samples_leaf,
                                           max_depth=self.max_depth)
            y_predict = estimator.predict(self.x)
            incorrect = y_predict != y

            # Error fraction
            estimator_error = np.average(incorrect, weights=sample_weight, axis=0)
    
            # Stop if the error is at least as bad as random guessing
            if estimator_error >= 1. - (1. / self.K):
                self.estimators_.pop(-1)
                self.values_.pop(-1)
                if len(self.estimators_) == 0:
                    raise ValueError('TreeClassification in AdaBoostClassification '
                                     'ensemble is worse than random, ensemble '
                                     'can not be fit.')
                return None
    
            # Boost weight using multi-class AdaBoost SAMME alg
            estimator_weight = (np.log((1. - estimator_error) / estimator_error) +
                np.log(self.K - 1.))
    
            self.values_.append(estimator_weight)
            sample_weight *= np.exp(estimator_weight * incorrect *
                                        ((sample_weight > 0) |
                                        (estimator_weight < 0)))
            sample_weight /= (np.sum(sample_weight) / self.n_samples)
            self.estimators_.append(estimator)

    def predict(self, X):
        preds = np.zeros((len(X), self.K))
        pred_codes = np.array([0., 1.])
        for estimator, value in zip(self.estimators_, self.values_):
            pred_labels = estimator.predict(X)
            pred_coding = pred_codes.take(np.arange(self.K) == pred_labels[:, np.newaxis])
            preds += value * pred_coding
        labels = np.argmax(preds, axis=1)
        return labels


if __name__ == "__main__":
#    samples = sampling([1, 2, 3, 4, 5], 10000, [4, 2, 2, 3, 6])
#    for i in range(1, 6):
#        print(np.mean(samples == i))
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import train_test_split
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    data = load_digits()
    x = data["data"]
    y = data["target"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    adaboost_full = AdaBoostClassification(100, min_samples_leaf=2, max_depth=-1)
    adaboost_full.fit(x_train, y_train)
    print("depth=-1|adaboost test accuracy: %s" 
          % np.mean(adaboost_full.predict(x_test) == y_test))
    
    sklearn_adaboost_full = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=2, max_depth=None),
                                          n_estimators=100,
                                          algorithm="SAMME")
    sklearn_adaboost_full.fit(x_train, y_train)
    print("depth=-1|sklearn adaboost test accuracy: %s" 
          % np.mean(sklearn_adaboost_full.predict(x_test) == y_test))
    
    adaboost = AdaBoostClassification(100, min_samples_leaf=2, max_depth=3)
    adaboost.fit(x_train, y_train)
    print("depth=3|adaboost test accuracy: %s" 
          % np.mean(adaboost.predict(x_test) == y_test))
    
    sklearn_adaboost = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=2, max_depth=3),
                                          n_estimators=100,
                                          algorithm="SAMME")
    sklearn_adaboost.fit(x_train, y_train)
    print("depth=3|sklearn adaboost test accuracy: %s" 
          % np.mean(sklearn_adaboost.predict(x_test) == y_test))
    