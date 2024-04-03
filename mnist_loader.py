import idx2numpy
import numpy as np

file = 'data/train-images-idx3-ubyte/train-images-idx3-ubyte'

train_img = idx2numpy.convert_from_file('data/train-images-idx3-ubyte/train-images-idx3-ubyte')
train_label = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte/train-labels-idx1-ubyte')

test_img = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_label = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

train_data = np.array(train_img[10000::])
#print(np.shape(train_data))

validation_data = np.array(train_img[::-1][50000::])
#print(np.shape(validation_data))

test_data = test_img


