//
// Created by Clytie on 2018/10/10.
//

#ifndef GBDT_UTILS_H
#define GBDT_UTILS_H

#include <vector>
#include <numeric>
#include <algorithm>

namespace utils {
    template <typename T>
    struct argvec {
        explicit argvec(const std::vector<T> & vec)
                : vec(vec) {}
        std::vector<T> vec;
        bool operator() (int i, int j) {return vec[i] < vec[j];}
    };

    template <typename T>
    std::vector<int> argsort(const std::vector<T> & v) {
        std::vector<int> idx(v.size());
        for (int i = 0; i < v.size(); ++i) {
            idx[i] = i;
        }

        argvec<T> compare(v);
        sort(idx.begin(), idx.end(), compare);
        return idx;
    }

    template <typename T>
    int argmax(const std::vector<T> & v) {
        if (v.size() <= 0) {
            return -1;
        }
        int max_index = 0;
        T max_value = v.front();
        for (int i = 1; i < v.size(); ++i) {
            if (v[i] > max_value) {
                max_value = v[i];
                max_index = i;
            }
        }
        return max_index;
    }
}

#endif //GBDT_UTILS_H
