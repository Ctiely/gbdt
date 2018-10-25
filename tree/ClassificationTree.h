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

#include "omp.h"
#include "utils.h"
using namespace std;

class ClassificationTree {
public:
    explicit ClassificationTree(const vector<vector<double> > & x,
                                const vector<int> & y,
                                const vector<double> & sample_weight,
                                int min_samples_leaf=2,
                                int max_depth=-1);
    ~ClassificationTree() = default;

    vector<int> predict(const vector<vector<double> > & x);
    vector<vector<double> > predict_proba(const vector<vector<double> > & x);
    //tree
    vector<int> Beg, End;
    vector<int> Pred, Cl, Cr, Spvb;
    vector<double> Ws;
    vector<bool> Leaf;
    vector<double> Spva;
    vector<int> Depth;
    //data
    const vector<vector<double> > x;
    const vector<int> y;
    int n_features;
    int nlevs;
private:
    void build(const vector<vector<double> > & x,
               const vector<int> & y,
               const vector<double> & sample_weight,
               int min_samples_leaf,
               int max_depth);
    pair<int, double> split(vector<int> & ta,
                           const vector<double> & tx,
                           const vector<int> & ty,
                           const vector<double> & sample_weight,
                           vector<double> & tpop,
                           double pno,
                           double pdo,
                           double crit0,
                           int tbeg,
                           int tend,
                           int nlevs);
    int predict(const vector<double> & x);
    vector<double> predict_proba(const vector<double> & x);
    
    vector<vector<double> > Pop;
};


#endif //GBDT_CLASSIFICATIONTREE_H
