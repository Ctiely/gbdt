#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 18:16:06 2018

@author: clytie
"""

import functools
import numpy as np

def compute_sse(y):
    return np.sum((y - np.mean(y)) ** 2)

def msplit(ta, tx, ty, crit0, tbeg, tend):
    critmax = crit0
    for i in range(tbeg, tend - 1):
        nc = ta[i]
        if tx[nc] == tx[ta[i + 1]]:
            continue
        left = ta[tbeg: i + 1]
        right = ta[i + 1: tend]
        left_y = ty[left]
        right_y = ty[right]
        left_sse = compute_sse(left_y)
        right_sse = compute_sse(right_y)
        after_sse = left_sse + right_sse
        if after_sse < critmax:
            nbest = i
            critmax = after_sse
    if critmax >= crit0:
        nbest = -1
    return [nbest, critmax]

def generate_tree(x, y, min_samples_leaf=2):
    x = x.T
    y = np.squeeze(y)
    pn = x.shape
    n = pn[1]
    p = pn[0]
    a = np.array(list(map(np.argsort, x)))
    Ns = []
    Beg = []
    End = []
    Leaf = []
    Cl = []
    Cr = []
    Spvb = []
    Spva = []
    Crit = []
    Pred = []
    Ns.append(n)
    Beg.append(0)
    End.append(n)
    Crit.append(compute_sse(y))
    Pred.append(np.mean(y))
    cur = 0
    kn = 0
    while cur <= kn:
        beg = Beg[cur]
        end = End[cur]
        mcrit0 = Crit[cur]
        mn = Ns[cur]
        if mcrit0 / mn < 1e-5:
            Leaf.append(True)
        else:
            msp = np.array(list(map(functools.partial(msplit, 
                                                      ty=y, 
                                                      crit0=mcrit0, 
                                                      tbeg=beg, 
                                                      tend=end), a, x)))
            ma = np.squeeze(msp[:,0].reshape(-1, 1), axis=1)
            if np.all(ma < 0):
                Leaf.append(True)
            else:
                sse = np.squeeze(msp[:,1])
                pvb = np.argmin(sse)
                pva = x[pvb][a[pvb][int(ma[pvb])]]
                left_ns = int(ma[pvb]) - beg + 1
                right_ns = mn - int(ma[pvb]) - 1 + beg
                if min(left_ns, right_ns) >= min_samples_leaf:
                    Leaf.append(False)
                    Spvb.append(pvb)
                    Spva.append(pva)
                    Cl.append(kn + 1)
                    Cr.append(kn + 2)
                    Ns.append(left_ns)
                    Beg.append(beg)
                    End.append(int(ma[pvb]) + 1)
                    left_y = y[a[pvb][Beg[kn + 1]: End[kn + 1]]]
                    Pred.append(np.mean(left_y))
                    Crit.append(compute_sse(left_y))
                    
                    Ns.append(right_ns)
                    Beg.append(int(ma[pvb]) + 1)
                    End.append(end)
                    right_y = y[a[pvb][Beg[kn + 2]: End[kn + 2]]]
                    Pred.append(np.mean(right_y))
                    Crit.append(compute_sse(right_y))
                    
                    kn += 2
                    tin = np.empty(n, dtype=int)
                    tin[a[pvb][beg: (int(ma[pvb]) + 1)]] = 1
                    tin[a[pvb][(int(ma[pvb]) + 1): end]] = 0
                    for i in range(p):
                        al = []
                        ar = []
                        if(i != pvb):
                            al = [ta for ta in a[i][beg: end] if tin[ta] == 1]
                            ar = [ta for ta in a[i][beg: end] if tin[ta] == 0]
                            a[i][beg: end] = np.hstack((al, ar))
                else:
                    Leaf.append(True)
        if Leaf[cur] == True:
            Spvb.append(-1)
            Spva.append(0)
            Cl.append(0)
            Cr.append(0)
        cur += 1
    mtree = {'v1cl': Cl,
             'v2cr': Cr,
             'v3spvb': Spvb,
             'v4spva': Spva,
             'v5ns': Ns,
             'v6pred': Pred,
             'v7leaf': Leaf,
             'v8beg': Beg,
             'v9end': End}
    return mtree
    
def _predict(ix, mytree):
    cur = 0
    Spvb = mytree['v3spvb']
    Spva = mytree['v4spva']
    Cl = mytree['v1cl']
    Cr = mytree['v2cr']
    Leaf = mytree['v7leaf']
    Pred = mytree['v6pred']
    while not Leaf[cur]:
        if ix[Spvb[cur]] <= Spva[cur]:
            cur = Cl[cur]
        else:
            cur = Cr[cur]
    return Pred[cur]

def predict(x, mytree):
    preds = []
    for ix in x:
        preds.append(_predict(ix, mytree))
    return preds
        
if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    
    data = load_boston()
    x = data["data"]
    y = data["target"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    tree = generate_tree(x_train, y_train)
    print(pd.DataFrame(tree))
    
    preds = predict(x_test, tree)
    preds = np.array(preds)
    
    mse = compute_sse(y_test) / len(y_test)
    print("mse: %s" % mse)
    
    mse_pred = np.mean((preds - y_test) ** 2)
    print("test mse: %s" % mse_pred)
    
    sktree = DecisionTreeRegressor()
    sktree.fit(x_train, y_train)
    preds_sk = sktree.predict(x_test)
    mse_pred_sk = np.mean((preds_sk - y_test) ** 2)
    print("sklearn test mse: %s" % mse_pred_sk)
