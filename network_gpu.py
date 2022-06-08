import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

from computation_graphs_gpu import *


class NetworkGpu:
    def __init__(self, input_shape=784, output_shape=10, size_hide_1=64, size_hide_2=16,
                 learning_rate=0.1, iter_per_epoch=10):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.size_hide_1 = size_hide_1
        self.size_hide_2 = size_hide_2
        self.learning_rate = learning_rate
        self.iter_per_epoch = iter_per_epoch

        self.network = {'W1': torch.rand(input_shape, self.size_hide_1, dtype=torch.float64).cuda() - 0.5,
                        'b1': torch.rand(self.size_hide_1, dtype=torch.float64).cuda() - 0.5,
                        'W2': torch.rand(self.size_hide_1, self.size_hide_2, dtype=torch.float64).cuda() - 0.5,
                        'b2': torch.rand(self.size_hide_2, dtype=torch.float64).cuda() - 0.5,
                        'W3': torch.rand(self.size_hide_2, output_shape, dtype=torch.float64).cuda() - 0.5,
                        'b3': torch.rand(output_shape, dtype=torch.float64).cuda() - 0.5
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

        self.train_log = torch.tensor([])
        self.correct_rate = 0

    def forward(self, inputs):
        """
        do forward, which means getting the predicts

        Parameters:
            inputs: test data with 2 dims

        Returns:
            got points for each kind
        """
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

    def backward(self, predicts, labels):
        """
        do backward

        Parameter:
            predicts : predict kinds, always from forward
            labels : correct kinds
        """
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
        self.network['b3'] -= torch.sum(diff_b3, 0) * self.learning_rate
        self.network['W2'] -= diff_w2 * self.learning_rate
        self.network['b2'] -= torch.sum(diff_b2, 0) * self.learning_rate
        self.network['W1'] -= diff_w1 * self.learning_rate
        self.network['b1'] -= torch.sum(diff_b1, 0) * self.learning_rate

    def train(self, train_data, train_label):
        """
        train for one epoch

        Parameters:
            train_data : data images with shape=(number, cols * rows)
            train_label : image labels with shape=(number,)
        """
        train_size = train_label.shape[0]
        batch_size = int(train_size / self.iter_per_epoch)
        for i in range(self.iter_per_epoch):
            train_batch = torch.reshape(train_data[i * batch_size: (i + 1) * batch_size], shape=(batch_size, 28 * 28))
            label_batch = train_label[i * batch_size: (i + 1) * batch_size]

            predict_before = self.forward(train_batch)
            self.backward(predict_before, label_batch)

        predict = self.forward(train_data.reshape(train_data.shape[0], 28 * 28))
        correct_rate = self.print_correct_rate(predict, train_label)
        self.train_log = torch.cat((self.train_log, torch.tensor([correct_rate])), 0)

    def show_train_log(self):
        """
        plot train log after training
        """
        x = range(0, self.train_log.numel())
        y = self.train_log
        plt.plot(x, y)
        plt.show()

    def get_correct_rate(self, predicts, labels):
        """
        get correct rate for predicts and labels

        Parameters:
            predicts: predict kinds, always from forward
            labels: correct kinds

        Returns:
            correct rate
        """
        predict = torch.argmax(predicts, 1)
        label = torch.argmax(labels, 1)
        corrects = torch.sum(predict == label)
        self.correct_rate = corrects / labels.shape[0]
        return self.correct_rate

    def print_correct_rate(self, predicts, labels):
        self.get_correct_rate(predicts, labels)
        print("correct rate: {rate:.2%}".format(rate=self.correct_rate))
        return self.correct_rate

    def save(self):
        """
        save the net to the new file:"test_correct_rate_{}.net".format(self.correct_rate)
        """
        save_file = open("test_correct_rate_{}.net".format(self.correct_rate), 'wb')

        net = {'input_shape': self.input_shape,
               'output_shape': self.output_shape,
               'size_hide_1': self.size_hide_1,
               'size_hide_2': self.size_hide_2,
               'learning_rate': self.learning_rate,
               'iter_per_epoch': self.iter_per_epoch,
               'train_log': self.train_log,
               'correct_rate': self.correct_rate,
               'network': self.network
               }

        pickle.dump(net, save_file)
        save_file.close()

    def load(self, load_path):
        """
        load net from load_path

        Parameters:
            load_path : exited file, which is saved from save()

        Returns:
            nothing but load the net
        """
        load_file = open(load_path, 'rb')
        net = pickle.load(load_file)
        self.input_shape = net['input_shape']
        self.output_shape = net['output_shape']
        self.size_hide_1 = net['size_hide_1']
        self.size_hide_2 = net['size_hide_2']
        self.learning_rate = net['learning_rate']
        self.iter_per_epoch = net['iter_per_epoch']
        self.train_log = net['train_log']
        self.correct_rate = net['correct_rate']
        self.network = net['network']
