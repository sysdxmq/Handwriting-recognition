

class Add:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, in_x, in_y):
        self.x = in_x
        self.y = in_y
        return self.x + self.y

    def backward(self, diff_out):
        dx = diff_out * 1
        dy = diff_out * 1
        return dx, dy


class Multiple:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, in_x, in_y):
        self.x = in_x
        self.y = in_y
        return self.x * self.y

    def backward(self, diff_out):
        dx = diff_out * self.y
        dy = diff_out * self.x
        return dx, dy
