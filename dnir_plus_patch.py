import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from keras.layers import Dense, Conv2D, Activation
from keras.models import Sequential
import os
from scipy.misc import imread, imresize
import skimage.io
import skimage.transform
from keras.layers.normalization import BatchNormalization
from math import floor
from keras import optimizers

# variable
patch_size = 40
stride = 10
filter_size = 3
patch_num = 225
patch_tnum = 530
patch_num_sqr = 15

# Load Test Imgs from Folder
cwd_tt = os.getcwd()
path_tt = cwd_tt + "/Train400"

# Print test imgs
print("%d files in %s" % (len(os.listdir(path_tt)), path_tt))

# Append Test Images and their Names to Lists
x_test = np.empty((9, 180, 180))
for f in range(9):
    x_test[f] = image.imread(path_tt + "/test_%03d.png" % (f + 1))
print ("%d images loaded" % (len(x_test)))
print("Type is %s" % (type(x_test)))

# Data normalization and reshaping
x_test = np.reshape(x_test, (len(x_test), 180, 180, 1))

# Add Noise
noise_factor = 25/255
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# load patches
patch_train_noisy = np.load('outfile_y.npy')
npatch_train = np.load('outfile_n.npy')
patch_test_noisy = np.load('outfile_ty.npy')
npatch_test = np.load('outfile_tn.npy')

# Model
model = Sequential()
model.add(Conv2D(64, (filter_size, filter_size), activation='relu', padding='same', input_shape=(patch_size, patch_size, 1)))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (filter_size, filter_size), activation=None, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(1, (filter_size, filter_size), activation='tanh', padding='same'))
# sgd = optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 0.001)
model.compile(optimizer='Adam', loss='mean_squared_error')

# Learning
model.fit(patch_train_noisy, npatch_train,
          epochs=10,
          batch_size=128,
          shuffle=True,
          validation_data=(patch_test_noisy, npatch_test))

# Reconstruction
decoded_imgs = model.predict(patch_test_noisy)
count = np.zeros((len(x_test), 180, 180, 1))
recover_test = np.empty((len(x_test), 180, 180, 1))
for m in range(len(x_test)):
    for n in range(patch_num):
        for p in range(patch_size):
            for t in range(patch_size):
               recover_test[m, floor(n / patch_num_sqr) * stride + p, (n % patch_num_sqr) * stride + t] \
                   += patch_test_noisy[m * patch_num + n, p, t] - decoded_imgs[m * patch_num + n, p, t]
               count[m, floor(n / patch_num_sqr) * stride + p, (n % patch_num_sqr) * stride + t] +=1

# Averaging
for m in range(len(x_test)):
    for row in range(180):
        for col in range(180):
            recover_test[m, row, col]/=count[m, row, col]

# Print 9 pics
n = 9
for i in range(n):
    # display original
    ax = plt.subplot(5, n, i + 1)
    plt.imshow(x_test[i].reshape(180, 180))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy input
    ax = plt.subplot(5, n, i + n + 1)
    plt.imshow(x_test_noisy[i].reshape(180, 180))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy patch
    ax = plt.subplot(5, n, i + 2 * n + 1)
    plt.imshow((npatch_test[i]).reshape(patch_size, patch_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display patch
    ax = plt.subplot(5, n, i + 3 * n + 1)
    plt.imshow((decoded_imgs[i]).reshape(patch_size, patch_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display clean estimation
    ax = plt.subplot(5, n, i + 4 * n + 1)
    plt.imshow((recover_test[i]).reshape(180, 180))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()