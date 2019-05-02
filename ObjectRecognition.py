import tensorflow
from tensorflow import keras
import matplotlib.pyplot
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import sys

print('Python: {}'.format(sys.version))
print('keras: {}',format(keras.__version__))

# Load The Data

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Let determine the dataset characteristics
print('Training Images: {}'.format(X_train.shape))
print('Testing Images: {}'.format(X_test.shape))

# A Single image
print(X_train[0].shape)

# Create a grid of 3*3 images

# for i in range(0, 9):
#    plt.subplot(330 + 1 + i)
#    img = X_train[i].transpose([1,2,0])
#    plt.imshow(img)

# show the plot
# plt.show()

# preprocesing the dataset

# random seed for reproducibility
seed = 6
np.random.seed(seed)

# load the data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# normalize the inputs from 0-255 to 0.0-1.0

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255.0
X_test = X_test/255.0

# class label shape

print(Y_train.shape)
print(Y_test[0])

# [6] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] one-hot vector

# hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_class = Y_test.shape[1]
print(num_class)

print(Y_train.shape)
print(Y_train[0])

# import layeres

import theano
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import SGD

# define the model function

def allcnn(weights = None):

    # define model type - Sequential
    model = Sequential()

    # add model layers
    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape = (32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding = 'valid'))

    # add global average pooling layer with softmax activation

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    # load the weights
    if weights:
        model.load_weights(weights)

    # return the model
    return model

# define hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# build model and define weights
model = allcnn()

# define optimizer and compile model
sgd = SGD(lr = learning_rate, decay = weight_decay, momentum = momentum, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# print model summary
print (model.summary())

# =============================================================

# MODEL TRAINING

# define additional training parameters

epochs = 360
batch_size = 32

# fit the model

model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = epochs, batch_size = batch_size, verbose = 1)

# =============================================================


#DEFINE HYPER PARAMETERS

learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# build the model and define weights 
weights = "all_cnn_weights_0.9088_0.4994.hdf5"
model = allcnn(weights)

# define optimizer and compile model
sgd = SGD(lr = learning_rate, decay = weight_decay, momentum = momentum, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# print model accuracy
print(model.summary())

# test the model with pretrained weights

scores = model.evaluate(X_test, Y_test, verbose = 1)
print('Accuracy: {}'.format(scores[1]))

# making a dictionary of class labels and names

classes =   range(0, 10)  

names = [ 'aeroplane',
          'automobile',
          'bird',
          'cat',
          'deer',
          'dog',
          'frog',
          'horse',
          'ship',
          'truck']
          
#zip the names and classes to make a dictionary of class labels

class_labels = dict(zip(classes, names))
print(class_labels)


# generate batch of 9 images

batch = X_test[100:109]
labels = rp.argmax(Y_test[100:109], axis = -1)

# make predictions

predictions = model.predict(batch, verbose = 1)

# print predictions

print(predictions)

# these are class probabilities, should sum to 1

for image in predictions:
    print(np.sum(image))
    
# use np.argmax() to convert class probabilities to class labels

class_result = np.argmax(predictions, axis = -1)
print(class_result)

# create a grid of 3*3 images

fig, axs = plt.subplot(3, 3, figsize = (15, 6))
fig.subplots_adjust(hspace = 1)
axs = axs.flatten()

for i, img in enumerate(batch):
    
    # determine label for each prediction, set title
    for key, value in class_labels.items():
        if class_result == key:
            title = 'Predictions: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)
            
        # plot the image
        
        axs[i].imshow(img.transpose([1, 2, 0]))
        
# show the plot

plt.show()

  








