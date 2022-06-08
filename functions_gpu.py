import torch


class Functions:
    @staticmethod
    def sigmoid(a):
        return 1 / (1 + torch.exp(-a))

    @staticmethod
    def softmax(a):
        max_a = torch.max(a, 1).values
        minus_a = (a.T - max_a).T
        exp_a = torch.exp(minus_a)
        exp_sum = torch.sum(exp_a, 1)
        out = (exp_a.T / exp_sum).T
        return out

    @staticmethod
    def relu(a):
        return torch.maximum(a, torch.ones_like(a))

    @staticmethod
    def cross_entropy_error(y, t):
        delta = 1e-7
        return -torch.sum(t * torch.log(y + delta))
