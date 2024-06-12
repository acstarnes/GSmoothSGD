# To run:
# 1. Specify any changes you have in the "Running hyperparameter search" section at the bottom
#      This includes which parameters you want to search over as well as the dataset and optimizer.
# 2. Activate the environment created from the requirements.txt file
# 3. Run `python hyperparameter_search.py`
#
# The output will be a csv file.


#################################################################################################
# Load packages
print('Loading packages')
import numpy as np
import pandas as pd

from tqdm import tqdm

from tensorflow.keras.datasets import mnist,cifar10

from tensorflow.keras import Input,Model,ops
from tensorflow.keras.layers import Conv2D,Dropout,AveragePooling2D,Dense,Flatten,Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

import math

import argparse
from itertools import product


#################################################################################################
# Defining functions
print('Defining functions')

pi = tf.constant(math.pi)
sigma = 1.

def load_environment(env_name = 'mnist'):
    
    if env_name == 'mnist':
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        x_train = x_train / 255
        x_test = x_test / 255
        y_train_sparse = to_categorical(y_train, num_classes=10)
        y_test_sparse = to_categorical(y_test, num_classes=10)
    elif env_name == 'cifar10':
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
        x_train = x_train / 255
        x_test = x_test / 255
        y_train_sparse = to_categorical(y_train, num_classes=10)
        y_test_sparse = to_categorical(y_test, num_classes=10)
    else:
        raise OSError(f'Could not load "{env_name}" data.')
    
    return x_train,y_train,x_test,y_test,y_train_sparse,y_test_sparse

def add_label_noise(
    train_labels,
    label_list = np.array(range(10), dtype='uint8'),
    noise = 0.05,
    seed = 16266
):
    '''
    labels = y_train
    label_list = list of all possible labels
    noise = percent of samples with noise added
    '''

    labels = np.array(train_labels, copy = True)

    rng = np.random.default_rng(seed=seed)

    n = int(len(labels) * noise) # number of samples to be modified

    samples_to_replace = rng.choice(len(labels), size = n, replace = False) # identify the indices of samples to replace

    replacements = [rng.choice(np.delete(label_list, labels[i])) for i in samples_to_replace] # randomly choose incorrect labels

    # Add noise to dataset
    for i in range(n):
        labels[samples_to_replace[i]] = replacements[i]
    
    labels_sparse = to_categorical(labels, num_classes=len(label_list))
    
    return labels,labels_sparse



def x_s(x):
    return x

def x2_s(x):
    return tf.math.square(x) + sigma ** 2 / 2

def tanh_s(x):
    return tf.math.tanh(x / tf.math.sqrt(1 + pi * sigma ** 2 / 4))

def tanh2_s(x):
    return 1. - (tf.math.sqrt(pi) * tf.math.exp(-4*tf.math.square(x) / (pi + 4 * sigma ** 2)) / (pi + 4 * sigma ** 2))

def relu_s(x):
    return (x / 2)*(1.+tf.math.erf(x / sigma)) + (sigma / (2 * tf.math.sqrt(pi))) * tf.math.exp(-tf.math.square(x) / (sigma ** 2))

def relu2_s(x):
    return (1/4)*(1+tf.math.erf(x / sigma))*(sigma ** 2 + 2 * tf.math.square(x)) + (1/(2*tf.math.sqrt(pi)))*sigma*x*tf.math.exp(-tf.math.square(x) / (sigma ** 2))

def matrix_norm(x):
    return tf.reduce_sum(tf.math.square(x))



class ReluRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self,strength = 0.01,scale = 1.):
        self.strength = strength
        self.scale = scale

    def __call__(self, x):
        # print('relu')
        return self.strength * self.scale * (ops.sum(relu2_s(x)) - ops.sum(ops.square(relu_s(x))))
    
    def get_config(self):
        return {'strength': self.strength, 'scale': self.scale}
    

class TanhRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self,strength = 0.01,scale = 1.):
        self.strength = strength
        self.scale = scale

    def __call__(self, x):
        return self.strength * self.scale * (ops.sum(tanh2_s(x)) - ops.sum(ops.square(tanh_s(x))))
    
    def get_config(self):
        return {'strength': self.strength, 'scale': self.scale}
    

class NormRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self,strength = 0.01,scale = 1.):
        self.strength = strength
        self.scale = scale

    def __call__(self, x):
        # print('norm')
        return self.strength * self.scale * ops.sum(ops.square(x))
    
    def get_config(self):
        return {'strength': self.strength, 'scale': self.scale}
    

class ConvolutionRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self,strength = 0.01,scale = 1.,layer = 4):
        self.strength = strength
        self.scale = scale
        if layer == 4:
            self.conv = l04_conv_reg
        elif layer == 7:
            self.conv = l07_conv_reg
        elif layer == 10:
            self.conv = l10_conv_reg
        else:
            raise ValueError('layer number not supported')

    def __call__(self, x):
        return self.strength * self.scale * ops.sum(ops.square(self.conv(x)))
    
    def get_config(self):
        return {'strength': self.strength, 'scale': self.scale, 'layer': self.layer}




def build_cnn(
    input_shape = (28,28,1),
    learning_rate = 0.01
):
    x00 = Input(input_shape, name = 'x0_input')

    l01 = Conv2D(32,4,1,'valid')
    l01_h = Activation('relu')
    l02 = AveragePooling2D(2)
    l03 = Flatten()
    l04 = Dense(128, activation = 'linear')
    l04_h = Activation('relu')
    l05 = Dense(10, activation = 'linear')

    x01 = l01(x00)
    x01_h = l01_h(x01)
    x02 = l02(x01_h)
    x03 = l03(x02)
    x04 = l04(x03)
    x04_h = l04_h(x04)
    y = l05(x04_h)

    # Model
    model = Model(
        inputs = x00,
        outputs = y
    )

    model.compile(
        loss = 'mse',
        optimzer = SGD(learning_rate),
        # optimizer = Adam(learning_rate),
        metrics = ['accuracy']
    )

    return model



def build_smooth_cnn(
    sigma = 1.,
    regularizer_strength = [0.01,0.01,0.01,0.01],
    input_shape = (28,28,1),
    learning_rate = 0.01
):
    x00 = Input(input_shape, name = 'x0_input')

    l01 = Conv2D(32,4,1,'valid', activity_regularizer=ReluRegularizer(regularizer_strength[0]))
    l01_h = Activation(relu_s)
    l02 = AveragePooling2D(2)
    l03 = Flatten(activity_regularizer=NormRegularizer(regularizer_strength[1], 0.5 * 128. * sigma ** 2))
    l04 = Dense(128, activation = 'linear', activity_regularizer=ReluRegularizer(regularizer_strength[2]), kernel_regularizer=NormRegularizer(regularizer_strength[1], 0.5 * sigma ** 2))
    l04_h = Activation(relu_s, activity_regularizer=NormRegularizer(regularizer_strength[3], 0.5 * 10 * sigma ** 2))
    l05 = Dense(10, activation = 'linear', kernel_regularizer=NormRegularizer(regularizer_strength[3], 0.5 * sigma ** 2))

    x01 = l01(x00)
    x01_h = l01_h(x01)
    x02 = l02(x01_h)
    x03 = l03(x02)
    x04 = l04(x03)
    x04_h = l04_h(x04)
    y = l05(x04_h)

    # Model
    model = Model(
        inputs = x00,
        outputs = y
    )

    model.compile(
        loss = 'mse',
        optimizer = SGD(learning_rate),
        # optimizer = Adam(learning_rate),
        metrics = ['accuracy']
    )

    return model



#################################################################################################
# Running hyperparameter search
print('Running hyperparameter search')

# parse the inputs
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='mnist',
                    help='Name of the dataset to use. Currently, can use "mnist" and "cifar10".')
parser.add_argument('-o', '--optimizer', default='adam',
                    help='Name of the optimizer to use (options: "adam","sgd").')
args = parser.parse_args()

dataset = args.dataset
optimizer = args.optimizer

x_train,y_train,x_test,y_test,y_train_sparse,y_test_sparse = load_environment(dataset)

pd.DataFrame(
    data = None,
    columns = ['accuracy', 'loss', 'val_accuracy', 'val_loss', 's', 'lr','l1', 'l3', 'l4', 'l4_h']
).to_csv(f'hyperparameter_search_{dataset}_{optimizer}.csv', index = False)

for reg in tqdm(list(product([1.,0.5,0.1],[0.001,0.0001,0.00001],[1e-5,1e-7],[1e-5,1e-7],[1e-5,1e-7],[1e-5,1e-7]))):
    print(reg)

    sigma = reg[0]
    sCNN = build_smooth_cnn(sigma = sigma, input_shape=x_train[0].shape, regularizer_strength=reg[2:], learning_rate = reg[1])

    early_stop = EarlyStopping(
        monitor = 'val_loss',
        patience = 2
    )

    sCNN.fit(
        x_train,
        y_train_sparse,
        epochs = 25,
        batch_size = 1,
        validation_data = (x_test,y_test_sparse),
        callbacks = [early_stop]
    )

    temp = sCNN.history.history
    temp_dict = {k:[temp[k][-1]] for k,v in temp.items()}
    temp_dict['s'] = reg[0]
    temp_dict['lr'] = reg[1]
    temp_dict['l1'] = reg[2]
    temp_dict['l3'] = reg[3]
    temp_dict['l4'] = reg[4]
    temp_dict['l4_h'] = reg[5]

    pd.DataFrame(
        temp_dict
    ).to_csv(f'hyperparameter_search_{dataset}_{optimizer}_batsz1.csv', index = False, mode = 'a', header = False)