import time
import torch
from mnist import Mnist
from network import Network
from network_gpu import NetworkGpu


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


def load(correct_rate):
    train_images, train_labels, test_images, test_labels = load_mnist()

    print("load test:\n")
    net_test = Network()
    net_test.load("test_correct_rate_{}.net".format(correct_rate))
    test_predict = net_test.forward(test_images.reshape(test_images.shape[0], 28 * 28))
    net_test.print_correct_rate(test_predict, test_labels)


def main():
    train_images, train_labels, test_images, test_labels = load_mnist()

    image_shape = 784
    train_epoch = 100

    start_train = time.time()

    net = Network(input_shape=image_shape, output_shape=10, iter_per_epoch=60)
    print("network training:")
    for i in range(train_epoch):
        net.train(train_images, train_labels)
    print("network training ended.\n")

    end_train = time.time()
    print("\ntrain time: {time:.4f}".format(time=end_train - start_train))

    print("test_image:")
    # net.show_train_log()

    test_predict = net.forward(test_images.reshape(test_images.shape[0], 28 * 28))
    correct_rate = net.print_correct_rate(test_predict, test_labels)

    # net.save()


def main_gpu():
    train_images, train_labels, test_images, test_labels = load_mnist()
    train_images = torch.from_numpy(train_images).cuda()
    train_labels = torch.from_numpy(train_labels).cuda()

    image_shape = 784
    train_epoch = 100

    start_train_gpu = time.time()

    net = NetworkGpu(input_shape=image_shape, output_shape=10, iter_per_epoch=60)
    print("network training:")
    for i in range(train_epoch):
        net.train(train_images, train_labels)
    print("network training ended.\n")

    end_train_gpu = time.time()
    print("\ngpu train time: {time:.4f}".format(time=end_train_gpu - start_train_gpu))

    print("test_image:")
    # net.show_train_log()

    test_images = torch.from_numpy(test_images).cuda()
    test_labels = torch.from_numpy(test_labels).cuda()
    test_predict = net.forward(test_images.reshape(test_images.shape[0], 28 * 28))
    net.print_correct_rate(test_predict, test_labels)


if __name__ == '__main__':
    # main()
    main_gpu()
