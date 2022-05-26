import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import math


class Network:
    def __init__(self, input_nodes=3, hidden_nodes=3, output_nodes=3, learning_rate=0.5):
        self.in_nodes = input_nodes
        self.hide_nodes = hidden_nodes
        self.out_nodes = output_nodes

        self.lr = learning_rate

        self.weight_in_hide = np.random.rand(self.hide_nodes, self.in_nodes) - 0.5
        self.weight_hide_out = np.random.rand(self.out_nodes, self.hide_nodes) - 0.5

    @staticmethod
    def activation_function(inputs):
        """sigmoid"""
        return scipy.special.expit(inputs)

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_inputs = np.dot(self.weight_in_hide, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weight_hide_out, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = final_outputs - targets
        hidden_errors = np.dot(self.weight_hide_out.T, output_errors)

        self.weight_hide_out -= self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                 np.transpose(hidden_outputs))
        self.weight_in_hide -= self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                np.transpose(inputs))
        ans = self.query(input_list)
        mse = sum(abs(ans - target_list)) / self.in_nodes
        print("mse =", mse)

    def train_until(self, input_list, target_list, target_mse=0.5):
        mse = 1.0
        while mse > target_mse:
            self.train(input_list, target_list)
            ans = self.query(input_list)
            mse = sum(abs(ans - target_list)) / self.in_nodes
            # print(ans, mse)
            # print("mse =", mse)

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2)
        hidden_inputs = np.dot(self.weight_in_hide, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weight_hide_out, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


class Test:
    def __init__(self, func=np.sin, start=-math.pi, end=math.pi, shape=100):
        self.shape = shape
        self.start = start
        self.end = end
        self.func = func
        self.x = np.linspace(self.start, self.end, self.shape)
        self.y = np.zeros_like(self.x)

    def run(self):
        self.set()
        # self.print()
        self.plot_original()

    def set(self):
        self.y = self.func(self.x)
        self.y += np.random.random(self.shape) / 2 - 0.25

    def print(self):
        print(self.shape)
        print(self.x)
        print(self.y)

    def train(self):
        pass

    def plot_original(self):
        x = np.linspace(self.start, self.end, self.shape)
        y = self.func(x)
        plt.plot(x, y)
        plt.scatter(x, self.y)
        plt.show()

    def plot_training_result(self):
        pass


def test_func(x):
    return np.sin(x)


if __name__ == "__main__":
    # train_func = np.frompyfunc(test_func, 1, 1)
    # test = Test(func=train_func, start=0, end=2 * math.pi, shape=40)
    # test.run()
    liner_func = lambda x: 0.5 * x
    input_shape = 1
    hidden_shape = 300
    output_shape = 1

    network = Network(input_nodes=input_shape, hidden_nodes=hidden_shape, output_nodes=output_shape)

    training_size = 101
    train_data = np.linspace(0, 1, training_size)
    train_target = liner_func(train_data)
    print("data =", train_data)
    print("target =", train_target)
    for data in train_data:
        ans_bf_train = network.query(data)
        print("ans before training =", ans_bf_train)

    for data, target in zip(train_data, train_target):
        network.train(input_list=data, target_list=target)

    for data, target in zip(train_data, train_target):
        ans_af_train = network.query(data)
        print("ans after training =", ans_af_train, "| target =", target)
