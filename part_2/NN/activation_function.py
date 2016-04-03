import numpy as np
from scipy.special import expit


class Logistic:
    def __init__(self):
        pass

    @staticmethod
    def value(x):
        return expit(x)

    @staticmethod
    def derivative(x):
        tmp = expit(x)
        return tmp * (1 - tmp)


class Tanh:
    def __init__(self):
        pass

    @staticmethod
    def value(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        tmp = np.tanh(x)
        return 1 - tmp**2


class Identity:
    def __init__(self):
        pass

    @staticmethod
    def value(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones_like(x)
