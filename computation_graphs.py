import numpy as np


class AddLayer:
    def __init__(self):
        pass

    @staticmethod
    def forward(in_x, in_y):
        out = in_x + in_y
        return out

    @staticmethod
    def backward(diff_out):
        dx = diff_out * 1
        dy = diff_out * 1
        return dx, dy


class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, in_x, in_y):
        self.x = in_x
        self.y = in_y
        return np.dot(self.x, self.y)

    def backward(self, diff_out):
        dx = np.dot(diff_out, np.transpose(self.y))
        dy = np.dot(np.transpose(self.x), diff_out)
        return dx, dy


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, in_x):
        self.mask = (in_x <= 0)
        out = in_x.copy()
        out[self.mask] = 0
        return out

    def backward(self, diff_out):
        diff_out[self.mask] = 0
        dx = diff_out
        return dx
