import numpy as np
import struct
from PIL import Image


class Mnist:
    def __init__(self,
                 data_path="./source/mnist/t10k-images.idx3-ubyte",
                 label_path="./source/mnist/t10k-labels.idx1-ubyte",
                 normalize=False):
        self.data_path = data_path
        self.label_path = label_path
        self.normalize = normalize
        pass

    def load_data(self, flatten=False):
        data_file = open(self.data_path, 'rb')
        magic, number, rows, cols = struct.unpack('>IIII', data_file.read(16))
        if magic != 2051:
            return []
        bytes_number = number * rows * cols
        images = np.array(struct.unpack('>' + str(bytes_number) + 'B', data_file.read()))

        data_file.close()

        if flatten:
            images = np.reshape(images, (number, rows * cols))
        else:
            images = np.reshape(images, (number, rows, cols))
        if self.normalize:
            images = images.astype('float64')
            images /= 255.0
        return images

    def load_label(self, one_hot=False):
        label_file = open(self.label_path, 'rb')
        magic, number = struct.unpack('>II', label_file.read(8))
        if magic != 2049:
            return []
        labels = np.array(struct.unpack('>' + str(number) + 'B', label_file.read()))

        label_file.close()

        if one_hot:
            labels_one_hot = np.zeros(shape=(number, 10))
            for i in range(0, number):
                labels_one_hot[i][labels[i]] = 1
            return labels_one_hot

        return labels


def image_show(image_data):
    img = Image.fromarray(image_data)
    img.show()


if __name__ == '__main__':
    data = Mnist()
    image = data.load_data()
    label = data.load_label()
    print(image.shape)
    image_show(image[0])
