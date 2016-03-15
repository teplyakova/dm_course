class Predicate:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold


class Tree:
    def __init__(self, left=None, right=None, info=None, class_distribution=None, is_leaf=False):
        self.left = left
        self.right = right
        self.info = info
        self.class_distribution = class_distribution
        self.is_leaf = is_leaf

    def test_feature(self, x):
        predicate = self.info
        if x[predicate.feature] <= predicate.threshold:
            return self.left
        else:
            return self.right
