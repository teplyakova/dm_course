import numpy as np
from tree import Tree, Predicate
from sklearn.preprocessing import LabelEncoder
from splitter import Splitter


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.encoder = LabelEncoder()
        self.n_samples = None
        self.n_classes = None
        self.n_features = None
        self.total_weight = None
        self.feature_importances_ = None
        self.tree_ = None

    def build_tree(self, X, y, sample_weight, class_distribution, level):
        # Need node weight for counting feature importances and probability of classes
        node_weight = np.sum(class_distribution)

        # node has only one class
        if np.unique(y).shape[0] == 1:
            proba = class_distribution / node_weight
            return Tree(info=proba, is_leaf=True)

        # node has less than min_samples_split samples
        if y.shape[0] < self.min_samples_split:
            proba = class_distribution / node_weight
            return Tree(info=proba, is_leaf=True)

        # node has less than 2*min_samples_leaf samples, so children would have less than min_sample_leaf samples
        if y.shape[0] < 2 * self.min_samples_leaf:
            proba = class_distribution / node_weight
            return Tree(info=proba, is_leaf=True)

        # tree has max_depth depth
        if self.max_depth is not None:
            if level == self.max_depth:
                proba = class_distribution / node_weight
                return Tree(info=proba, is_leaf=True)

        splitter = Splitter()
        feature, threshold, split_pos, index = splitter.find_best_split(X, y, sample_weight,
                                                                        class_distribution, self.min_samples_leaf)
        if feature is None:
            proba = class_distribution / node_weight
            return Tree(info=proba, is_leaf=True)

        gain = splitter.impurity_gain
        left_distribution = splitter.left_distribution
        right_distribution = splitter.right_distribution

        self.feature_importances_[feature] += gain * (float(node_weight) / self.total_weight)
        X = X[index]
        y = y[index]
        sample_weight = sample_weight[index]

        left_tree = self.build_tree(X[0:split_pos], y[0:split_pos], sample_weight[0:split_pos],
                                    left_distribution, level+1)
        right_tree = self.build_tree(X[split_pos:], y[split_pos:], sample_weight[split_pos:],
                                     right_distribution, level+1)
        return Tree(left_tree, right_tree, Predicate(feature, threshold), class_distribution)

    def fit(self, X, y, sample_weight=None):
        self.n_samples = y.shape[0]
        self.n_features = X.shape[1]

        # Equal weights
        if sample_weight is None:
            sample_weight = np.full(self.n_samples, 1. / self.n_samples)

        # Encode classes as 0, 1, ... , n-1
        y = self.encoder.fit_transform(y)
        class_distribution = np.bincount(y, sample_weight)
        self.n_classes = self.encoder.classes_.shape[0]

        # Need total weight to count feature importances
        self.total_weight = np.sum(class_distribution)
        self.feature_importances_ = np.zeros(X.shape[1])

        # Build tree
        self.tree_ = self.build_tree(X, y, sample_weight, class_distribution, level=0)

    def predict_sample_proba(self, sample, cur_node):
        if cur_node.is_leaf:
            return cur_node.info
        else:
            next_node = cur_node.test_feature(sample)
            return self.predict_sample_proba(sample, next_node)

    def predict_proba(self, X):
        if self.tree_ is None:
            return None
        probas = np.zeros((X.shape[0], self.n_classes))
        for i in xrange(X.shape[0]):
            probas[i] = self.predict_sample_proba(X[i], self.tree_)
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        classes = np.argmax(probas, axis=1)
        return self.encoder.inverse_transform(classes)
