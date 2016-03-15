import numpy as np
from criterion import GiniCriterion

FEATURE_THRESHOLD = 1e-7


class Splitter:
    def __init__(self):
        self.class_distribution = None
        self.impurity_gain = None
        self.left_distribution = None
        self.right_distribution = None
        self.min_samples_leaf = None

    def split_feature(self, values, y, sample_weight):
        best_gain = None
        best_split_pos = None
        best_left_distribution = None

        p = 1
        start = 0
        end = y.shape[0]

        criterion = GiniCriterion(y, sample_weight, self.class_distribution)

        while p < end:
            while p + 1 < end and values[p] <= values[p-1] + FEATURE_THRESHOLD:
                p += 1
            cur_pos = p
            # check for min_samples_leaf in left and right child
            if cur_pos - start < self.min_samples_leaf or end - cur_pos < self.min_samples_leaf:
                p += 1
                continue
            criterion.update(cur_pos)
            gain = criterion.impurity_gain()
            if gain > best_gain:
                best_gain = gain
                best_split_pos = cur_pos
                best_left_distribution = np.copy(criterion.left_distribution)
            p += 1

        return best_gain, best_split_pos, best_left_distribution

    def find_best_split(self, X, y, sample_weight, class_distribution, min_samples_leaf):
        self.class_distribution = class_distribution
        n_features = X.shape[1]
        self.min_samples_leaf = min_samples_leaf

        best_split_pos = None
        best_feature = None
        sorted_index = None
        threshold = None

        for feature in xrange(n_features):
            values = X[:, feature]
            index = np.argsort(values)
            sorted_values = values[index]
            sorted_y = y[index]
            sorted_sample_weight = sample_weight[index]

            gain, split_pos, left_distribution = self.split_feature(sorted_values, sorted_y, sorted_sample_weight)
            if gain > self.impurity_gain:
                self.impurity_gain = gain
                best_split_pos = split_pos
                best_feature = feature
                sorted_index = np.copy(index)
                threshold = (sorted_values[best_split_pos] + sorted_values[best_split_pos-1]) / 2.0
                self.left_distribution = np.copy(left_distribution)

        if self.impurity_gain is not None:
            self.right_distribution = self.class_distribution - self.left_distribution

        return best_feature, threshold, best_split_pos, sorted_index
