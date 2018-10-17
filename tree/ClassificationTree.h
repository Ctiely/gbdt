//
// Created by Clytie on 2018/10/10.
//

#ifndef GBDT_CLASSIFICATIONTREE_H
#define GBDT_CLASSIFICATIONTREE_H

#include <cmath>
#include <vector>
#include <numeric>
#include <utility>
#include <cassert>

#include "utils.h"
using namespace std;

class ClassificationTree {
public:
    explicit ClassificationTree(const vector<vector<float> > & x,
                                const vector<int> & y,
                                const vector<float> & sample_weight,
                                int min_samples_leaf=2,
                                int max_depth=-1);
    ~ClassificationTree() = default;

    vector<int> predict(const vector<vector<float> > & x);
    vector<vector<float> > predict_proba(const vector<vector<float> > & x);
    //tree
    vector<int> Beg, End;
    vector<int> Pred, Cl, Cr, Spvb;
    vector<float> Ws;
    vector<bool> Leaf;
    vector<float> Spva;
    vector<int> Depth;
    //data
    const vector<vector<float> > x;
    const vector<int> y;
    int n_features;
    int nlevs;
private:
    void build(const vector<vector<float> > & x,
               const vector<int> & y,
               const vector<float> & sample_weight,
               int min_samples_leaf,
               int max_depth);
    pair<int, float> split(vector<int> & ta,
                           const vector<float> & tx,
                           const vector<int> & ty,
                           const vector<float> & sample_weight,
                           vector<float> & tpop,
                           float pno,
                           float pdo,
                           float crit0,
                           int tbeg,
                           int tend,
                           int nlevs);
    int predict(const vector<float> & x);
    vector<float> predict_proba(const vector<float> & x);
    
    vector<vector<float> > Pop;
};


#endif //GBDT_CLASSIFICATIONTREE_H
