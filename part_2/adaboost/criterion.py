import numpy as np


class GiniCriterion:
    def __init__(self, y, sample_weight, class_distribution):
        self.y = y
        self.sample_weight = sample_weight

        # total weight of all samples
        weight = np.sum(class_distribution)
        self.node_weight = weight
        self.left_weight = 0.0
        self.right_weight = weight

        self.pos = 0

        self.distribution = class_distribution
        self.left_distribution = np.zeros_like(class_distribution)
        self.right_distribution = np.copy(class_distribution)

        self.impurity = 1.0 - np.sum(self.distribution * self.distribution) / weight**2

    def update(self, new_pos):
        # Update statistics up to new_pos
        for p in xrange(self.pos, new_pos):
            cl = self.y[p]
            w = self.sample_weight[p]
            self.left_distribution[cl] += w
            self.left_weight += w
            self.right_distribution[cl] -= w
            self.right_weight -= w

        self.pos = new_pos

    def node_impurity(self):
        return self.impurity

    def children_impurity(self):
        gini_left = 1.0 - np.sum(self.left_distribution * self.left_distribution) / self.left_weight**2
        gini_right = 1.0 - np.sum(self.right_distribution * self.right_distribution) / self.right_weight**2
        return gini_left, gini_right

    def impurity_gain(self):
        left, right = self.children_impurity()
        left_ratio = self.left_weight / self.node_weight
        right_ratio = self.right_weight / self.node_weight
        return self.node_impurity() - left_ratio * left - right_ratio * right
