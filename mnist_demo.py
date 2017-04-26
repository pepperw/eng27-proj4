# MNIST handwritten digit recognition - data file loading demo
# Written by Matt Zucker, April 2017

import numpy as np
import gzip
import struct
import cv2
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D
from keras.layers.core import Flatten
from keras.utils.np_utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
import time
IMAGE_SIZE = 28

######################################################################
# Read a 32-bit int from a file or a stream

def read_int(f):
    buf = f.read(4)
    data = struct.unpack('>i', buf)
    return data[0]

######################################################################
# Open a regular file or a gzipped file to decompress on-the-fly

def open_maybe_gz(filename, mode='rb'):

    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

######################################################################
# Read the MNIST data from an images file or a labels file. The file
# formats are documented at http://yann.lecun.com/exdb/mnist/

def read_mnist(images_file, labels_file):

    images = open_maybe_gz(images_file)

    imagic = read_int(images)
    assert(imagic == 2051)
    icount = read_int(images)
    rows = read_int(images)
    cols = read_int(images)
    assert(rows == IMAGE_SIZE and cols == IMAGE_SIZE)

    print 'reading', icount, 'images of', rows, 'rows by', cols, 'cols.'

    labels = open_maybe_gz(labels_file)

    lmagic = read_int(labels)
    assert(lmagic == 2049)
    lcount = read_int(labels)

    print 'reading', lcount, 'labels.'

    assert(icount == lcount)

    image_array = np.fromstring(images.read(icount*rows*cols),
                                dtype=np.uint8).reshape((icount,rows,cols))

    label_array = np.fromstring(labels.read(lcount),
                                dtype=np.uint8).reshape((icount))

    return image_array, label_array

def multi_neural_network(x,y,x_test,y_test):
    x = x * (2.0/255.0) - 1.0
    x = x.reshape(len(x), 28, 28, 1)
    x_test = x_test * (2.0/255.0) - 1.0
    x_test = x_test.reshape(len(x_test), 28, 28, 1)

    y_train = to_categorical(y, 10)
    y_test = to_categorical(y_test, 10)
    neural_net = Sequential()
    rows, cols = x.shape[1:3]
    neural_net.add(Conv2D(5, (5, 5), strides=(1,1), activation='relu', input_shape=(rows, cols, 1)))
    neural_net.add(MaxPooling2D(pool_size = (2,2)))
    neural_net.add(Flatten())
    neural_net.add(Dense(100, activation='sigmoid'))
    neural_net.add(Dense(10, activation='softmax'))
    neural_net.summary()


    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    neural_net.compile(optimizer= sgd, loss="categorical_crossentropy", metrics=['accuracy'])
    start1 = time.time()
    history = neural_net.fit(x, y_train, verbose=1, validation_data=(x_test, y_test), epochs=10)
    end1 = time.time()

    ## test models
    start2 = time.time()
    loss, accuracy = neural_net.evaluate(x_test, y_test)
    end2 = time.time()
    print
    print "accuracy: ", accuracy
    print "training time in seconds: ", end1-start1
    print "testing time in seconds: ", end2-start2
    """
    conv2D: 20 features
    accuracy:  0.989
    training time in seconds:  144.918672085
    testing time in seconds:  0.621148824692

    5 features:
    accuracy:  0.9866
    training time in seconds:  115.518862963
    testing time in seconds:  0.358344078064


    """


def pca_knn(x,y,x_test,y_test):
    #PCA phase
    x = x.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
    x_test = x_test.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
    start = time.time()
    mean, eigenv = cv2.PCACompute(data = x, mean = np.array([]),retainedVariance = 0.95)
    print len(eigenv)

    new_x = cv2.PCAProject(x, mean, eigenv[:51])

    # knn phase
    new_x_test = cv2.PCAProject(x_test, mean, eigenv[:51])
    neighbors = KNeighborsClassifier(n_neighbors = 3)

    neighbors.fit(new_x,y)
    start1 = time.time()
    accuracy = neighbors.score(new_x_test,y_test)
    end1 = time.time()
    print
    print "accuracy: ", accuracy
    print "training time in seconds: ", start1-start
    print "testing time in seconds: ", end1-start1
    """
    pca dimension: 153
    accuracy:  0.9723
    training time in seconds:  32.8320491314
    testing time in seconds:  108.31241107

    pca dimension: 50
    accuracy:  0.9752
    training time in seconds:  28.3141319752
    testing time in seconds:  34.398319006



    """



######################################################################
# Show use of the MNIST data set:


def main():

    # Read images and labels. This is reading the 10k-element test set
    # (you can also use the other pair of filenames to get the
    # 60k-element training set).
    images, labels = read_mnist('MNIST_data/t10k-images-idx3-ubyte.gz',
                                'MNIST_data/t10k-labels-idx1-ubyte.gz')
    train_images, train_labels = read_mnist('MNIST_data/train-images-idx3-ubyte.gz',
                                'MNIST_data/train-labels-idx1-ubyte.gz')


    # This is a nice way to reshape and rescale the MNIST data
    # (e.g. to feed to PCA, Neural Net, etc.) It converts the data to
    # 32-bit floating point, and then recenters it to be in the [-1,
    # 1] range.

    #classifier_input = images.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
    #classifier_input = images * (2.0/255.0) - 1.0
    #classifier_input = classifier_input.reshape(len(images), 28, 28, 1)

    #train_input = train_images.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
    #train_input = train_images * (2.0/255.0) - 1.0
    #train_input = train_input.reshape(len(train_input), 28, 28, 1)

    ##train the multi-layer neural network
    #multi_neural_network(train_images,train_labels,images,labels)
    pca_knn(train_images,train_labels,images,labels)






    ##################################################
    # Now just display some stuff:
"""
    print 'images has datatype {}, shape {}, and ranges from {} to {}'.format(
        images.dtype, images.shape, images.min(), images.max())

    print 'input has datatype {}, shape {}, and ranges from {} to {}'.format(
        classifier_input.dtype, classifier_input.shape,
        classifier_input.min(), classifier_input.max())

    for i, image in enumerate(images):
        display = cv2.resize(image, (8*IMAGE_SIZE, 8*IMAGE_SIZE),
                             interpolation=cv2.INTER_NEAREST)

        print 'image {} has label {}'.format(i, labels[i])
        cv2.imshow('win', display)
        while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass
"""
######################################################################

if __name__ == '__main__':
    main()
