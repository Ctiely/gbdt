//
// Created by Clytie on 2018/10/10.
//

#ifndef GBDT_CLASSIFICATIONTREE_H
#define GBDT_CLASSIFICATIONTREE_H

#include <vector>
#include <numeric>
#include <utility>

#include "utils.h"
using namespace std;

class ClassificationTree {
public:
    explicit ClassificationTree(const vector<vector<float> > & x, const vector<int> & y, int min_samples_leaf=2);
    ~ClassificationTree() = default;

    vector<int> predict(const vector<vector<float> > & x);

    const vector<vector<float> > x;
    const vector<int> y;
    int n_features;
    int nlevs;
private:
    void build(const vector<vector<float> > & x, const vector<int> & y, int min_samples_leaf);
    pair<int, float> split(vector<int> & ta,
                           const vector<float> & tx,
                           const vector<int> & ty,
                           vector<float> & tpop,
                           float pno,
                           float pdo,
                           float crit0,
                           int tbeg,
                           int tend,
                           int nlevs);
    int predict(const vector<float> & x);

    vector<int> Pred, Cl, Cr, Spvb;
    vector<bool> Leaf;
    vector<float> Spva;
};


#endif //GBDT_CLASSIFICATIONTREE_H
