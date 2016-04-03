import numpy as np


class MSELoss:
    def __init__(self):
        pass

    @staticmethod
    def count(t, y):
        return 0.5 * np.sum((t - y) * (t - y))

    @staticmethod
    def derivative(t, y):
        return y - t
