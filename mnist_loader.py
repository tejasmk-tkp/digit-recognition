import idx2numpy
import numpy as np
import cv2

file = 'data/train-images-idx3-ubyte/train-images-idx3-ubyte'

train_img = idx2numpy.convert_from_file('data/train-images-idx3-ubyte/train-images-idx3-ubyte')
train_label = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte/train-labels-idx1-ubyte')

test_img = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_label = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

'''
print(np.shape(label))
print(label)

cv2.imshow("Image", img[1])

cv2.waitKey(0)

if KeyboardInterrupt:
    print("Closed by User!")
    cv2.destroyAllWindows()
'''

train_data = np.array(train_img[10000::])
print(np.shape(train_data))

validation_data = np.array(train_img[::-1][10000::])
print(np.shape(validation_data))

'''
#cv2.imshow('Image 1', train_data[0])
cv2.imshow('Image 2', validation_data[0])

cv2.waitKey(0)

if KeyboardInterrupt:
    print("Closed by User!")
    cv2.destroyAllWindows()
'''

'''
if train_data[0].all() == validation_data[0].all():
    print(train_data[0])
    print(" ")
    print(validation_data[0])

    print("It's the same")
'''

test_data = test_img
