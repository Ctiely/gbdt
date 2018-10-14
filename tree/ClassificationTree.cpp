//
// Created by Clytie on 2018/10/10.
//

#include "ClassificationTree.h"

ClassificationTree::ClassificationTree(const vector<vector<float> > & x, const vector<int> & y, int min_samples_leaf)
        : x(x), y(y), Cl(), Cr(), Spvb(), Spva(), Leaf(), Pred(), nlevs(0) {
    n_features = (int)x.size();
    build(x, y, min_samples_leaf);
}

void ClassificationTree::build(const vector<vector<float> > & x, const vector<int> & y, int min_samples_leaf) {
    nlevs = *max_element(y.begin(), y.end()) + 1;
    auto p = x.size();
    auto n = x.front().size();
    vector<vector<int> > a(p);
    for (auto i = 0; i < p; ++i) {
        a[i] = utils::argsort(x[i]);
    }
    vector<int> Ns, Beg, End;
    vector<vector<float> > Pop;
    Ns.push_back((int)n);
    Beg.push_back(0);
    End.push_back((int)n);
    vector<float> mpop((unsigned long)nlevs);
    for (const int iy : y) {
        mpop[iy] += 1;
    }
    Pop.push_back(mpop);
    Pred.push_back(utils::argmax(mpop));
    int cur = 0, kn = 0;
    while (cur <= kn) {
        int beg = Beg[cur];
        int end = End[cur];
        vector<float> tpop = Pop[cur];
        float mpno = 0;
        for (const auto & pop : tpop) {
            mpno += pop * pop;
        }
        float mpdo = accumulate(tpop.begin(), tpop.end(), 0.0f);
        float mcrit0 = mpno / mpdo;
        int mn = Ns[cur];
        if (mcrit0 / mn > 1 - 1e-4) {
            Leaf.push_back(true);
        } else {
            vector<int> ma(p);
            vector<float> mgini(p);
            bool have_split = false;
            for (int i = 0; i < p; ++i) {
                auto msplit = split(a[i], x[i], y, tpop, mpno, mpdo, mcrit0, beg, end, nlevs);
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
                float pva = x[pvb][a[pvb][ma[pvb]]];
                int left_ns = ma[pvb] - beg + 1;
                int right_ns = mn - ma[pvb] - 1 + beg;
                if (min(left_ns, right_ns) >= min_samples_leaf) {
                    Leaf.push_back(false);
                    Spvb.push_back(pvb);
                    Spva.push_back(pva);
                    Cl.push_back(kn + 1);
                    Cr.push_back(kn + 2);
                    Ns.push_back(left_ns);
                    Beg.push_back(beg);
                    End.push_back(ma[pvb] + 1);
                    vector<float> ltpop((unsigned long)nlevs);
                    for (int i = Beg[kn + 1]; i < End[kn + 1]; ++i) {
                        ltpop[y[a[pvb][i]]] += 1;
                    }
                    Pop.push_back(ltpop);
                    Pred.push_back(utils::argmax(ltpop));
                    Ns.push_back(right_ns);
                    Beg.push_back(ma[pvb] + 1);
                    End.push_back(end);
                    vector<float> rtpop((unsigned long)nlevs);
                    for (int i = 0; i < nlevs; ++i) {
                        rtpop[i] = tpop[i] - ltpop[i];
                    }
                    Pop.push_back(rtpop);
                    Pred.push_back(utils::argmax(rtpop));
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

pair<int, float> ClassificationTree::split(vector<int> & ta,
                                           const vector<float> & tx,
                                           const vector<int> & ty,
                                           vector<float> & tpop,
                                           float pno,
                                           float pdo,
                                           float crit0,
                                           int tbeg,
                                           int tend,
                                           int nlevs) {
    int nbest = -1;
    float critmax = crit0;
    float rrn = pno;
    float rrd = pdo;
    float rln = 0.0f;
    float rld = 0.0f;
    vector<float> wl((unsigned long)nlevs);
    vector<float> wr = tpop;
    for (int i = tbeg; i < tend - 1; ++i) {
        int nc = ta[i];
        int k = ty[nc];
        rln = rln + (2 * wl[k] + 1);
        rrn = rrn + (-2 * wr[k] + 1);
        rld = rld + 1;
        rrd = rrd - 1;
        wl[k] = wl[k] + 1;
        wr[k] = wr[k] - 1;
        if (tx[nc] < tx[ta[i + 1]]) {
            float crit = rln / rld + rrn / rrd;
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

int ClassificationTree::predict(const vector<float> & x) {
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

vector<int> ClassificationTree::predict(const vector<vector<float> > & x) {
    vector<int> preds;
    for (const auto & ix : x) {
        preds.push_back(predict(ix));
    }
    return preds;
}
