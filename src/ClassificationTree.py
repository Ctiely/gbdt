#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 21:00:17 2018

@author: clytie
"""

import numpy as np
from GBDT import TreeClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier

data = load_digits()
x = data["data"]
y = data["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
tree = TreeClassification(x_train, y_train, min_samples_leaf=1)

preds = tree.predict(x_test)
print("test accuracy: %s" % np.mean(preds == y_test))

sktree = DecisionTreeClassifier()
sktree.fit(x_train, y_train)

preds_sk = sktree.predict(x_test)
print("test accuracy: %s" % np.mean(preds_sk == y_test))

if __name__ == "___main__":
    pass