# Generates the data for the SVRG experiments
#
# To run:
# 1. Specify any changes you have in the "Building Heatmap" section at the bottom
#      This includes which parameters you want to search over as well as the dataset and optimizer.
# 2. Activate the environment created from the requirements.txt file
# 3. Run `python svrg_noise_heatmap.py`
#
# The output will be a csv file.

#################################################################################################
# Load packagesimport numpy as np
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

#################################################################################################
# Defining functions
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


def add_input_noise(
    train,
    noise = 0.05,
    seed = 16266
):
    '''
    train = x_train
    noise = stdev of gaussian
    '''

    noisy_train = np.array(train, copy = True)

    rng = np.random.default_rng(seed=seed)

    noise = rng.normal(scale = noise, size = train.shape)

    noisy_train = noisy_train + noise
    
    return noisy_train


def x_s(x):
    return x

def x2_s(x):
    return tf.math.square(x) + sigma ** 2 / 2

def tanh_s(x):
    return tf.math.tanh(x / tf.math.sqrt(1 + pi * sigma ** 2 / 4))

def tanh2_s(x):
    return 1. - (tf.math.sqrt(pi) * tf.math.exp(-4*tf.math.square(x) / (pi + 4 * sigma ** 2)) / (pi + 4 * sigma ** 2))

def relu_s(x):
    return (x / 2)*(1.+tf.math.erf(x / sigma)) + (sigma / 2) * tf.math.exp(-tf.math.square(x) / (sigma ** 2))

def relu2_s(x):
    return tf.math.abs((1/4)*(1+tf.math.erf(x / sigma))*(sigma ** 2 + 2 * tf.math.square(x)) + (1/(2*tf.math.sqrt(pi)))*sigma*x*tf.math.exp(-tf.math.square(x) / (sigma ** 2)))

def sqroot_relu2_s(x):
    return tf.math.sqrt(tf.math.abs((1/4)*(1+tf.math.erf(x / sigma))*(sigma ** 2 + 2 * tf.math.square(x)) + (1/(2*tf.math.sqrt(pi)))*sigma*x*tf.math.exp(-tf.math.square(x) / (sigma ** 2))))

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
    

class NormRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self,strength = 0.01,scale = 1.):
        self.strength = strength
        self.scale = scale

    def __call__(self, x):
        # print('norm')
        return self.strength * self.scale * ops.sum(ops.square(x))
    
    def get_config(self):
        return {'strength': self.strength, 'scale': self.scale}



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
        optimizer = SGD(learning_rate),
        metrics = ['accuracy']
    )

    return model


def build_smooth_cnn(
    sigma = 1.,
    # regularizer_strength = [0.01,0.01,0.01,0.01,0.01,0.01,0.01],
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
        # optimizer = 'adam',
        # optimizer = Adam(0.001),
        metrics = ['accuracy']
    )

    return model





##################################################################################################
# Building Heatmap
sx_train,sy_train,x_test,y_test,sy_train_sparse,y_test_sparse = load_environment('mnist')
from itertools import product

# Specify these (at least)
inner_steps = 10000
outer_steps = 6
steps = 60000

for noise in tqdm(product([0,0.25,0.5,0.75,1.],[0,0.1,0.2,0.3,0.4])):

    filename = 'x' + str(int(100*noise[0])) + '_y' + str(int(100*noise[1]))

    x_train = add_input_noise(sx_train, noise = noise[0])
    y_train,y_train_sparse = add_label_noise(sy_train, noise = noise[1])

    def gradient(model,sample):

        with tf.GradientTape() as tape:
            loss = model.compute_loss(x_train[sample],y_train_sparse[sample],model(x_train[sample]))
        
        return tape.gradient(loss, model.trainable_variables)

    def svrg_saves(
        model = build_cnn(),
        outer_model = build_cnn(),
        out_steps = 10,
        inner_steps = 10,
        learning_rate = 0.001,
        name = 'svrg'
    ):
        
        variance_rng = np.random.default_rng(seed=7376)
        variance_sample = variance_rng.choice(len(x_train), 100)
        outer_rng = np.random.default_rng(seed=894615)
        sample_rng = np.random.default_rng(seed=950647)

        pd.DataFrame(
            data = None,
            columns = ['step', 'val_accuracy', 'type']
        ).to_csv(f'./svrg_logs/svrg_training_{name}lr02.csv', index = False)

        pd.DataFrame(
            data = None,
            columns = range(100)
        ).to_csv(f'./svrg_logs/grad_norms_{name}lr02.csv', index = False)

        outer_model.set_weights(model.get_weights())

        for s in tqdm(range(out_steps), desc = 'outer'):
        
            # Compute $\widetilde{m}_{\sigma_s}$
            sample = outer_rng.choice(len(x_train), 4000)
            control = gradient(outer_model,sample)

            # $x_0=\widetilde{x}$
            # local_model.set_weights(model.get_weights())

            for t in tqdm(range(inner_steps), desc = 'inner'):
                # Save
                if t % 100 == 0:
                    acc = tf.keras.metrics.CategoricalAccuracy()(model(x_test),y_test_sparse).numpy()
                    pd.DataFrame(
                        data = [[inner_steps*s+t,acc,name]],
                        columns = ['step', 'val_accuracy', 'type']
                    ).to_csv(f'./svrg_logs/svrg_training_{name}.csv', index = False, mode = 'a', header = False)

                    grad_norms = np.zeros(100, dtype='float32')
                    index = -1
                    for i in variance_sample:
                        index += 1
                        model_grad = gradient(model,[i])
                        outer_grad = gradient(outer_model,[i])
                        grad_norms[index] = ops.sum([ops.sum(ops.square(ops.add(ops.subtract(l[0],l[1]),l[2]))) for l in zip(model_grad,outer_grad,control)]).numpy()
                    pd.DataFrame(
                        data = [grad_norms],
                        columns = range(100)
                    ).to_csv(f'./svrg_logs/grad_norms_{name}.csv', index = False, mode = 'a', header = False)

                # SVRG
                sample = sample_rng.choice(len(x_train), 1)
                model_grad = gradient(model,sample)
                outer_grad = gradient(outer_model,sample)
                grad = [ops.multiply(learning_rate,ops.add(ops.subtract(l[0],l[1]),l[2])) for l in zip(model_grad,outer_grad,control)]
                model.set_weights([ops.subtract(l[0],l[1]) for l in zip(model.get_weights(),grad)])
            
            outer_model.set_weights(model.get_weights())
        
        return model


    def sgd_saves(
        model = build_cnn(),
        steps = 10,
        learning_rate = 0.1,
        name = 'sgd'
    ):
        
        variance_rng = np.random.default_rng(seed=7376)
        variance_sample = variance_rng.choice(len(x_train), 100)
        sample_rng = np.random.default_rng(seed=950647)

        acc = tf.keras.metrics.CategoricalAccuracy()(model(x_test),y_test_sparse).numpy()
        pd.DataFrame(
            data = None,
            columns = ['step', 'val_accuracy', 'type']
        ).to_csv(f'./svrg_logs/svrg_training_{name}.csv', index = False)

        pd.DataFrame(
            data = None,
            columns = range(1000)
        ).to_csv(f'./svrg_logs/grad_norms_{name}.csv', index = False)

        for t in tqdm(range(steps)):

            # Save
            if t % 100 == 0:
                acc = tf.keras.metrics.CategoricalAccuracy()(model(x_test),y_test_sparse).numpy()
                # print(acc)
                pd.DataFrame(
                    data = [[t,acc,name]],
                    columns = ['step', 'val_accuracy', 'type']
                ).to_csv(f'./svrg_logs/svrg_training_{name}.csv', index = False, mode = 'a', header = False)
                grad_norms = [ops.sum([ops.sum(ops.square(l)) for l in gradient(model,[i])]).numpy() for i in variance_sample]
                pd.DataFrame(
                    data = [grad_norms],
                    columns = range(100)
                ).to_csv(f'./svrg_logs/grad_norms_{name}.csv', index = False, mode = 'a', header = False)
            
            # SGD
            sample = sample_rng.choice(len(x_train), 1)
            grad = [ops.multiply(learning_rate,l) for l in gradient(model,sample)]
            model.set_weights([ops.subtract(l[0],l[1]) for l in zip(model.get_weights(),grad)])
        
        return model

    sigma = 0.1
    scnn_init = build_smooth_cnn(
        sigma=0.1,
        input_shape=x_train[0].shape,
        regularizer_strength=[1e-7,1e-7,1e-7,1e-7]
    )

    cnn_sgd = build_cnn(input_shape=x_train[0].shape)
    cnn_sgd.set_weights(scnn_init.get_weights())

    cnn_sgd = sgd_saves(
        model = cnn_sgd,
        steps = 5000,
        learning_rate = 0.01,
        name = 'sgd' + filename
    )

    scnn_sgd = build_smooth_cnn(sigma=0.1,input_shape=x_train[0].shape,regularizer_strength=[1e-7,1e-7,1e-5,1e-5])
    scnn_sgd.set_weights(scnn_init.get_weights())

    sgd_saves(
        model = scnn_sgd,
        steps = steps,
        learning_rate = 0.01,
        name = 'gsmoothsgd' + filename
    )

    cnn_svrg = build_cnn(input_shape=x_train[0].shape)
    cnn_svrg.set_weights(scnn_init.get_weights())

    cnn_svrg = svrg_saves(
        cnn_svrg,
        build_cnn(input_shape=x_train[0].shape),
        out_steps = outer_steps,
        inner_steps = inner_steps,
        learning_rate = 0.01,
        name = 'svrg' + filename
    )

    scnn_svrg = build_smooth_cnn(sigma=0.1,input_shape=x_train[0].shape,regularizer_strength=[1e-13,1e-13,1e-13,1e-13])
    scnn_svrg.set_weights(scnn_init.get_weights())

    scnn_svrg = svrg_saves(
        scnn_svrg,
        build_smooth_cnn(sigma=0.1,input_shape=x_train[0].shape,regularizer_strength=[1e-13,1e-13,1e-13,1e-13]),
        out_steps = outer_steps,
        inner_steps = inner_steps,
        learning_rate = 0.01,
        name = 'gsmoothsvrg_insmooth' + filename
    )

