import matplotlib.pyplot as plt
import time
from matplotlib.image import imread
from PIL import Image
from mnist import Mnist
from computation_graphs import *


class Network:
    def __init__(self, input_shape=2, output_shape=3, size_hide_1=64, size_hide_2=16,
                 learning_rate=0.1, iter_per_epoch=10,
                 activity_func=Functions.softmax, identity_func=Functions.softmax):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.iter_per_epoch = iter_per_epoch

        self.size_hide_1 = size_hide_1
        self.size_hide_2 = size_hide_2
        self.network = {'W1': np.random.rand(input_shape, self.size_hide_1) - 0.5,
                        'b1': np.random.rand(self.size_hide_1) - 0.5,
                        'W2': np.random.rand(self.size_hide_1, self.size_hide_2) - 0.5,
                        'b2': np.random.rand(self.size_hide_2) - 0.5,
                        'W3': np.random.rand(self.size_hide_2, output_shape) - 0.5,
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
                       'R3': ReLULayer(),
                       'SL': SoftmaxWithLoss()
                       }

        self.activity_func = activity_func
        self.identity_func = identity_func

        self.train_log = []

    def activity_function(self, inputs):
        return self.activity_func(inputs)

    def identity_function(self, inputs):
        return self.identity_func(inputs)

    def forward(self, inputs):
        m1 = self.graphs['M1'].forward(inputs, self.network['W1'])
        a1 = self.graphs['A1'].forward(m1, self.network['b1'])
        r1 = self.graphs['R1'].forward(a1)
        m2 = self.graphs['M2'].forward(r1, self.network['W2'])
        a2 = self.graphs['A2'].forward(m2, self.network['b2'])
        r2 = self.graphs['R2'].forward(a2)
        m3 = self.graphs['M3'].forward(r2, self.network['W3'])
        a3 = self.graphs['A3'].forward(m3, self.network['b3'])
        r3 = self.graphs['R3'].forward(a3)
        # print("r3 = {}".format(r3))
        return r3

    def backward(self, predicts, labels):
        self.graphs['SL'].forward(predicts, labels)
        diff_sl = self.graphs['SL'].backward()
        diff_r3 = self.graphs['R3'].backward(diff_sl)
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

    def train(self, train_data, train_label):
        train_size = train_label.shape[0]
        batch_size = int(train_size / self.iter_per_epoch)
        for i in range(self.iter_per_epoch):
            train_batch = np.reshape(train_data[i * batch_size: (i + 1) * batch_size], newshape=(batch_size, 28 * 28))
            label_batch = train_label[i * batch_size: (i + 1) * batch_size]

            predict_before = self.forward(train_batch)
            self.backward(predict_before, label_batch)

        predict = self.forward(train_data.reshape(train_data.shape[0], 28*28))
        correct_rate = self.print_correct_rate(predict, train_label)
        self.train_log.append(correct_rate)

    def show_train_log(self):
        log = np.array(self.train_log)
        x = range(0, log.shape[0])
        y = self.train_log
        plt.plot(x, y)
        plt.show()

    @staticmethod
    def get_correct_rate(predicts, labels):
        predict = np.argmax(predicts, axis=1)
        label = np.argmax(labels, axis=1)
        corrects = np.sum(predict == label)
        correct_rate = corrects / labels.shape[0]
        return correct_rate

    def print_correct_rate(self, predicts, labels):
        correct_rate = self.get_correct_rate(predicts, labels)
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
    train_labels = train_data.load_label(one_hot=True)

    test_data = Mnist(data_path=mnist_path_test_data, label_path=mnist_path_test_label)
    test_images = test_data.load_data(flatten=False, normalize=True)
    test_labels = test_data.load_label(one_hot=True)

    return [train_images, train_labels, test_images, test_labels]


def main():
    train_images, train_labels, test_images, test_labels = load_mnist()

    image_shape = 784
    train_epoch = 100

    net = Network(input_shape=image_shape, output_shape=10, iter_per_epoch=60)
    for i in range(train_epoch):
        net.train(train_images, train_labels)

    net.show_train_log()

    print("\n")
    test = net.forward(test_images.reshape(test_images.shape[0], 28*28))
    net.print_correct_rate(test, test_labels)


if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    print("total run time: {time:.4f}".format(time=end - start))
