import numpy as np


class Functions:
    @staticmethod
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def softmax(a):
        max_a = np.max(a, axis=1)
        minus_a = (a.T - max_a).T
        exp_a = np.exp(minus_a)
        exp_sum = np.sum(exp_a, axis=1)
        out = (exp_a.T / exp_sum).T
        return out

    @staticmethod
    def relu(a):
        return np.maximum(a, 0)

    @staticmethod
    def cross_entropy_error(y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))
