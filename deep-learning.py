import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.image import imread
from PIL import Image
from mnist import Mnist
from functions import Functions
from computation_graphs import *


class Network:
    def __init__(self, input_shape=2, output_shape=3, activity_func=Functions.softmax, identity_func=Functions.softmax):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.network = {'W1': np.random.rand(input_shape, 3)-0.5,
                        'b1': np.random.rand(3)-0.5,
                        'W2': np.random.rand(3, 4)-0.5,
                        'b2': np.random.rand(4)-0.5,
                        'W3': np.random.rand(4, output_shape)-0.5,
                        'b3': np.random.rand(output_shape)-0.5
                        }

        self.activity_func = activity_func
        self.identity_func = identity_func

    def activity_function(self, inputs):
        return self.activity_func(inputs)

    def identity_function(self, inputs):
        return self.identity_func(inputs)

    def forward(self, inputs):
        a1 = np.dot(inputs, self.network['W1']) + self.network['b1']
        z1 = self.activity_function(a1)
        a2 = np.dot(z1, self.network['W2']) + self.network['b2']
        z2 = self.activity_function(a2)
        a3 = np.dot(z2, self.network['W3']) + self.network['b3']
        outputs = self.identity_function(a3)
        return outputs

    def backward(self):
        pass

    @staticmethod
    def print_correct_rate(predicts, labels):
        corrects = np.sum(predicts == labels)
        correct_rate = corrects/labels.shape[0]
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
    y = net.forward(image_batch)
    predict = np.argmax(y, axis=1)
    Network.print_correct_rate(predicts=predict, labels=label_batch)


if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    print("total run time: {time:.4f}".format(time=end-start))
