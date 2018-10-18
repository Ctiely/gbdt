//
// Created by Clytie on 2018/10/10.
//

#include "ClassificationTree.h"

ClassificationTree::ClassificationTree(const vector<vector<double> > & x,
                                       const vector<int> & y,
                                       const vector<double> & sample_weight,
                                       int min_samples_leaf,
                                       int max_depth)
        : x(x), y(y) {
    n_features = (int)x.size();
    assert(y.size() == sample_weight.size());
    max_depth = (max_depth == -1) ? (int)log2(x.front().size()) : max_depth;
    build(x, y, sample_weight, min_samples_leaf, max_depth);
}

void ClassificationTree::build(const vector<vector<double> > & x,
                               const vector<int> & y,
                               const vector<double> & sample_weight,
                               int min_samples_leaf,
                               int max_depth) {
    nlevs = *max_element(y.begin(), y.end()) + 1;
    auto p = x.size();
    auto n = x.front().size();
    vector<vector<int> > a(p);
    for (auto i = 0; i < p; ++i) {
        a[i] = utils::argsort(x[i]);
    }
    Ws.push_back(accumulate(sample_weight.begin(), sample_weight.end(), 0.0));
    Beg.push_back(0);
    End.push_back((int)n);
    vector<double> mpop((unsigned long)nlevs);
    for (int i = 0; i < y.size(); ++i) {
        mpop[y[i]] += sample_weight[i];
    }
    Depth.push_back(0);
    Pop.push_back(mpop);
    Pred.push_back(utils::argmax(mpop));
    int cur = 0, kn = 0;
    while (cur <= kn) {
        int beg = Beg[cur];
        int end = End[cur];
        vector<double> tpop = Pop[cur];
        double mpno = 0;
        for (const auto & pop : tpop) {
            mpno += pop * pop;
        }
        double mpdo = Ws[cur];
        double mcrit0 = mpno / mpdo;
        int depth = Depth[cur];
        if (mcrit0 / mpdo > 1 - 1e-4 || depth >= max_depth) {
            Leaf.push_back(true);
        } else {
            vector<int> ma(p);
            vector<double> mgini(p);
            bool have_split = false;
            for (int i = 0; i < p; ++i) {
                auto msplit = split(a[i], x[i], y, sample_weight, tpop, mpno, mpdo, mcrit0, beg, end, nlevs);
                ma[i] = msplit.first;
                mgini[i] = msplit.second;
                if (msplit.first >= 0) {
                    have_split = true;
                }
            }
            if (!have_split) {
                Leaf.push_back(true);
            } else {
                int pvb = utils::argmax(mgini);
                double pva = x[pvb][a[pvb][ma[pvb]]];
                int left_ns = ma[pvb] - beg + 1;
                int right_ns = end - ma[pvb];
                if (min(left_ns, right_ns) >= min_samples_leaf) {
                    int left_end = ma[pvb] + 1;
                    vector<double> ltpop((unsigned long)nlevs);
                    for (int i = beg; i < left_end; ++i) {
                        ltpop[y[a[pvb][i]]] += sample_weight[a[pvb][i]];
                    }
                    double left_w = accumulate(ltpop.begin(), ltpop.end(), 0.0);
                    double right_w = mpdo - left_w;
                    if (min(left_w, right_w) <= 1e-4) { //哇,这一行贼关键
                        Leaf.push_back(true);
                    } else {
                        Leaf.push_back(false);
                        Spvb.push_back(pvb);
                        Spva.push_back(pva);
                        Cl.push_back(kn + 1);
                        Cr.push_back(kn + 2);
                        //left
                        Beg.push_back(beg);
                        End.push_back(left_end);
                        Ws.push_back(left_w);
                        Pop.push_back(ltpop);
                        Pred.push_back(utils::argmax(ltpop));
                        Depth.push_back(depth + 1);
                        //right
                        Beg.push_back(ma[pvb] + 1);
                        End.push_back(end);
                        Ws.push_back(right_w);
                        vector<double> rtpop((unsigned long)nlevs);
                        for (int i = 0; i < nlevs; ++i) {
                            rtpop[i] = tpop[i] - ltpop[i];
                        }
                        Pop.push_back(rtpop);
                        Pred.push_back(utils::argmax(rtpop));
                        Depth.push_back(depth + 1);
                        
                        kn += 2;
                        
                        vector<int> tin(n);
                        for (int i = beg; i < ma[pvb] + 1; ++i) {
                            tin[a[pvb][i]] = 1;
                        }
                        for (int i = ma[pvb] + 1; i < end; ++i) {
                            tin[a[pvb][i]] = 0;
                        }
                        for (int i = 0; i < p; ++i) {
                            if (i != pvb) {
                                vector<int> al, ar;
                                for (int j = beg; j < end; ++j) {
                                    int ta = a[i][j];
                                    if (tin[ta]) {
                                        al.push_back(ta);
                                    } else {
                                        ar.push_back(ta);
                                    }
                                }
                                for (int k = 0; k < al.size(); ++k) {
                                    a[i][beg + k] = al[k];
                                }
                                for (int k = 0; k < ar.size(); ++k) {
                                    a[i][beg + al.size() + k] = ar[k];
                                }
                            }
                        }
                    }
                } else {
                    Leaf.push_back(true);
                }
            }
        }
        if (Leaf[cur]) {
            Spvb.push_back(-1);
            Spva.push_back(0);
            Cl.push_back(0);
            Cr.push_back(0);
        }
        ++cur;
    }
}

pair<int, double> ClassificationTree::split(vector<int> & ta,
                                           const vector<double> & tx,
                                           const vector<int> & ty,
                                           const vector<double> & sample_weight,
                                           vector<double> & tpop,
                                           double pno,
                                           double pdo,
                                           double crit0,
                                           int tbeg,
                                           int tend,
                                           int nlevs) {
    int nbest = -1;
    double critmax = crit0;
    double rrn = pno;
    double rrd = pdo;
    double rln = 0;
    double rld = 0;
    vector<double> wl((unsigned long)nlevs);
    vector<double> wr = tpop;
    for (int i = tbeg; i < tend - 1; ++i) {
        int nc = ta[i];
        int k = ty[nc];
        rln += (2 * wl[k] + sample_weight[nc]) * sample_weight[nc];
        rrn += (-2 * wr[k] + sample_weight[nc]) * sample_weight[nc];
        rld += sample_weight[nc];
        rrd -= sample_weight[nc];
        wl[k] = wl[k] + sample_weight[nc];
        wr[k] = wr[k] - sample_weight[nc];
        if (tx[nc] < tx[ta[i + 1]]) {
            double crit = rln / rld + rrn / rrd;
            if (crit > critmax) {
                nbest = i;
                critmax = crit;
            }
        }
    }
    if (critmax <= crit0) {
        nbest = -1;
    }
    return make_pair(nbest, critmax);
}

int ClassificationTree::predict(const vector<double> & x) {
    int cur = 0;
    while (!Leaf[cur]) {
        if (x[Spvb[cur]] <= Spva[cur]) {
            cur = Cl[cur];
        } else {
            cur = Cr[cur];
        }
    }
    return Pred[cur];
}

vector<int> ClassificationTree::predict(const vector<vector<double> > & x) {
    vector<int> preds;
    for (const auto & ix : x) {
        preds.push_back(predict(ix));
    }
    return preds;
}

vector<double> ClassificationTree::predict_proba(const vector<double> & x) {
    int cur = 0;
    while (!Leaf[cur]) {
        if (x[Spvb[cur]] <= Spva[cur]) {
            cur = Cl[cur];
        } else {
            cur = Cr[cur];
        }
    }
    vector<double> probs = Pop[cur];
    double cur_w = Ws[cur];
    for (int i = 0; i < probs.size(); ++i) {
        probs[i] /= cur_w;
    }
    return probs;
}

vector<vector<double> > ClassificationTree::predict_proba(const vector<vector<double> > & x) {
    vector<vector<double> > probs;
    for (const auto & ix : x) {
        probs.push_back(predict_proba(ix));
    }
    return probs;
}
