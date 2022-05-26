import numpy as np


class Functions:
    @staticmethod
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def softmax(a):
        max_a = np.max(a, axis=0)
        exp_a = np.exp(a - max_a)
        exp_sum = np.sum(exp_a, axis=1)
        return (exp_a.T / exp_sum).T

    @staticmethod
    def relu(a):
        return np.maximum(a, 0)
