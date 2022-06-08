import torch
from functions_gpu import Functions
import time


class AddLayer:
    def __init__(self):
        pass

    @staticmethod
    def forward(in_x, in_y):
        out = in_x + in_y
        return out

    @staticmethod
    def backward(diff_out):
        dx = diff_out
        dy = diff_out
        return dx, dy


class MultiLayer:
    def __init__(self):
        torch.cuda.set_device(0)
        self.x = None
        self.y = None

    def forward(self, in_x, in_y):
        self.x = in_x
        self.y = in_y
        return self.x @ self.y

    def backward(self, diff_out):
        dx = diff_out @ self.y.T
        dy = self.x.T @ diff_out
        return dx, dy


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, in_x):
        self.mask = (in_x <= 0)
        out = in_x.clone()
        out[self.mask] = 0
        return out

    def backward(self, diff_out):
        diff_out[self.mask] = 0
        dx = diff_out
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        torch.cuda.set_device(0)
        self.loss = None  # 损失
        self.y = None  # softmax输出
        self.t = None  # 监督数据（one-hot vector）
        pass

    def forward(self, in_x, in_t):
        self.t = in_t
        self.y = Functions.softmax(in_x)
        self.loss = Functions.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, diff_out=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


def main():
    shape = (10000, 10000)
    x = torch.ones(shape)
    y = torch.ones(shape)
    layer = MultiLayer()
    print(layer.forward(x, y))


if __name__ == '__main__':
    start = time.time()

    # main()
    a = torch.randn(3, 3)
    print(a)
    print(torch.max(a, 1).values)

    end = time.time()
    print("\ntotal run time: {time:.4f}".format(time=end - start))