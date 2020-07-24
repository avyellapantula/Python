"""
Task at hand:
We will now illustrate how to create a CNN in Keras
to solve a classification task using the MNIST dataset
and achieve human-like performance.
"""

"""
Import the libs
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from matplotlib import pyplot as plt

"""
Loading the Data
"""
batch_size = 128
epochs = 2

# we have 10 classes
# digits from 0-9
num_classes = 10

# now we load the data,
# splitting the data into
# training and testing sets
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# make the data suitable
# recall we're working with image data

# input image dimensions
img_rows, img_cols = X_train[0].shape
# Reshaping the data to use it in our network
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
# Scaling the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Let's check one of the datapoints:
plt.imshow(X_test[1][..., 0], cmap='Greys')
plt.axis('off')
plt.show()