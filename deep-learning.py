import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.image import imread
from PIL import Image
from mnist import Mnist
from functions import Functions
from computation_graphs import *


class Network:
    def __init__(self, input_shape=2, output_shape=3, learning_rate=0.1, activity_func=Functions.softmax, identity_func=Functions.softmax):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate

        self.network = {'W1': np.random.rand(input_shape, 3) - 0.5,
                        'b1': np.random.rand(3) - 0.5,
                        'W2': np.random.rand(3, 4) - 0.5,
                        'b2': np.random.rand(4) - 0.5,
                        'W3': np.random.rand(4, output_shape) - 0.5,
                        'b3': np.random.rand(output_shape) - 0.5
                        }

        self.graphs = {'M1': MultiLayer(),
                       'A1': AddLayer(),
                       'R1': ReLULayer(),
                       'M2': MultiLayer(),
                       'A2': AddLayer(),
                       'R2': ReLULayer(),
                       'M3': MultiLayer(),
                       'A3': AddLayer(),
                       'R3': ReLULayer()}

        self.activity_func = activity_func
        self.identity_func = identity_func

    def activity_function(self, inputs):
        return self.activity_func(inputs)

    def identity_function(self, inputs):
        return self.identity_func(inputs)

    def graph_forward(self, inputs):
        m1 = self.graphs['M1'].forward(inputs, self.network['W1'])
        a1 = self.graphs['A1'].forward(m1, self.network['b1'])
        r1 = self.graphs['R1'].forward(a1)
        m2 = self.graphs['M2'].forward(r1, self.network['W2'])
        a2 = self.graphs['A2'].forward(m2, self.network['b2'])
        r2 = self.graphs['R2'].forward(a2)
        m3 = self.graphs['M3'].forward(r2, self.network['W3'])
        a3 = self.graphs['A3'].forward(m3, self.network['b3'])
        r3 = self.graphs['R3'].forward(a3)
        return r3

    def forward(self, inputs):
        a1 = np.dot(inputs, self.network['W1']) + self.network['b1']
        z1 = self.activity_function(a1)
        a2 = np.dot(z1, self.network['W2']) + self.network['b2']
        z2 = self.activity_function(a2)
        a3 = np.dot(z2, self.network['W3']) + self.network['b3']
        outputs = self.identity_function(a3)
        return outputs

    def backward(self, predicts, labels):
        diff = predicts - labels
        diff_out = diff * diff
        diff_r3 = self.graphs['R3'].backward(diff_out)
        diff_a3, diff_b3 = self.graphs['A3'].backward(diff_r3)
        diff_m3, diff_w3 = self.graphs['M3'].backward(diff_a3)
        diff_r2 = self.graphs['R2'].backward(diff_m3)
        diff_a2, diff_b2 = self.graphs['A2'].backward(diff_r2)
        diff_m2, diff_w2 = self.graphs['M2'].backward(diff_a2)
        diff_r1 = self.graphs['R1'].backward(diff_m2)
        diff_a1, diff_b1 = self.graphs['A1'].backward(diff_r1)
        diff_m1, diff_w1 = self.graphs['M1'].backward(diff_a1)

        self.network['W3'] -= diff_w3 * self.learning_rate
        self.network['b3'] -= np.sum(diff_b3, axis=0) * self.learning_rate
        self.network['W2'] -= diff_w2 * self.learning_rate
        self.network['b2'] -= np.sum(diff_b2, axis=0) * self.learning_rate
        self.network['W1'] -= diff_w1 * self.learning_rate
        self.network['b1'] -= np.sum(diff_b1, axis=0) * self.learning_rate

    @staticmethod
    def print_correct_rate(predicts, labels):
        corrects = np.sum(predicts == labels)
        correct_rate = corrects / labels.shape[0]
        print("correct rate: {rate:.2%}".format(rate=correct_rate))
        return correct_rate


def lena_show():
    img = imread("./source/lena512.bmp")
    plt.imshow(img)
    plt.show()


def image_show(image_data):
    img = Image.fromarray(image_data)
    img.show()


def load_mnist():
    mnist_path_train_data = "./source/mnist/train-images.idx3-ubyte"
    mnist_path_train_label = "./source/mnist/train-labels.idx1-ubyte"
    mnist_path_test_data = "./source/mnist/t10k-images.idx3-ubyte"
    mnist_path_test_label = "./source/mnist/t10k-labels.idx1-ubyte"

    train_data = Mnist(data_path=mnist_path_train_data, label_path=mnist_path_train_label)
    train_images = train_data.load_data(flatten=False, normalize=True)
    train_labels = train_data.load_label(one_hot=False)

    test_data = Mnist(data_path=mnist_path_test_data, label_path=mnist_path_test_label)
    test_images = test_data.load_data(flatten=False, normalize=True)
    test_labels = test_data.load_label(one_hot=False)

    return [train_images, train_labels, test_images, test_labels]


def main():
    train_images, train_labels, test_images, test_labels = load_mnist()

    image_shape = 784
    batch_size = 1000

    # random_choice = np.random.choice(train_labels.shape[0], batch_size)

    image_batch = np.reshape(train_images[0:batch_size], [batch_size, image_shape])
    label_batch = np.array(train_labels[0:batch_size])
    net = Network(input_shape=image_shape, output_shape=10)
    y = net.graph_forward(image_batch)
    predict = np.argmax(y, axis=1)
    Network.print_correct_rate(predicts=predict, labels=label_batch)

    labels_one_hot = np.zeros(shape=(batch_size, 10))
    for i in range(0, batch_size):
        labels_one_hot[i][label_batch[i]] = 1

    net.backward(y, labels_one_hot)
    z = net.graph_forward(image_batch)
    predict_2 = np.argmax(z, axis=1)
    Network.print_correct_rate(predicts=predict_2, labels=label_batch)


if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    print("total run time: {time:.4f}".format(time=end - start))
