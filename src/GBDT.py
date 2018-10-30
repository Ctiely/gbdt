#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:00:30 2018

@author: clytie
"""
    
if __name__ == "__main__":
    import numpy as np
    from GBDT import GBDTClassifier
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    
    data = load_digits()
    x = data["data"]
    y = data["target"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    gbdt = GBDTClassifier(n_estimators=100, max_depth=3)
    gbdt.fit(x_train, y_train)
    
    print("learning_rate=0.1|gbdt test accuracy: %s" 
          % np.mean(gbdt.predict(x_test) == y_test))
    
    skgbdt = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    skgbdt.fit(x_train, y_train)
    
    print("learning_rate=0.1|sklearn gbdt test accuracy: %s" 
          % np.mean(skgbdt.predict(x_test) == y_test))
    
    print("=" * 50)
    
    gbdt = GBDTClassifier(learning_rate=1, n_estimators=100, max_depth=3)
    gbdt.fit(x_train, y_train)
    
    print("learning_rate=1|gbdt test accuracy: %s" 
          % np.mean(gbdt.predict(x_test) == y_test))
    
    skgbdt = GradientBoostingClassifier(learning_rate=1, n_estimators=100, max_depth=3)
    skgbdt.fit(x_train, y_train)
    
    print("learning_rate=1|sklearn gbdt test accuracy: %s" 
          % np.mean(skgbdt.predict(x_test) == y_test))
