##########################################################################################
# Imports
##########################################################################################
import argparse
import yaml

import numpy as np
import pandas as pd

from tqdm import tqdm

from tensorflow.keras.datasets import mnist,cifar10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

from sklearn.metrics import accuracy_score


##########################################################################################
# Load argument
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--config',
    default='config',
    help='Name of the config file'
)
args = parser.parse_args()
config_file = args.config

config = yaml.safe_load(open(f'./configs/{config_file}.yml'))

tau = 1


##########################################################################################
# Load data
##########################################################################################
def load_environment(env_name = config['env_name']):
    
    if env_name == 'mnist':
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
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
    
    # Optional: Subset the data, because 60000 training images is too many!
    sample = np.random.choice(range(len(x_train)),10000)
    x_train = x_train[sample]
    y_train = y_train[sample]
    y_train_sparse = y_train_sparse[sample]
    
    return x_train,y_train,x_test,y_test,y_train_sparse,y_test_sparse


##########################################################################################
# Build model
##########################################################################################
def build_model(
    input = load_environment(config['env_name'])[0][0].shape,
    layers = config['layers'],
    output = len(load_environment(config['env_name'])[5][0])
):
    
    tf.random.set_seed(config['seed'])
    
    model = Sequential()
    model.add(Flatten(input_shape = input))

    for l in layers:
        model.add(Dense(l, activation = 'relu'))

    model.add(Dense(output, activation = 'softmax'))
    model.compile()
    
    return model

##########################################################################################
# Losses
##########################################################################################
def f(
    x = build_model(), # current version of model
    x_dataset = load_environment(config['env_name'])[0],
    y_dataset = load_environment(config['env_name'])[1],
    sample = range(len(load_environment(config['env_name'])[0])), # samples to be used in loss computation
):
    
    # x_tr = x_dataset[sample] + np.random.normal(scale = 0.01, size = x_dataset[sample].shape)
    # preds = x(x_tr)
    preds = x(x_dataset[sample])
    return CategoricalCrossentropy()(preds,y_dataset[sample])

def misclassification_rate(
    x = build_model(), # current version of model
    x_dataset = load_environment(config['env_name'])[0],
    y_dataset = load_environment(config['env_name'])[1],
    sample = range(len(load_environment(config['env_name'])[0])), # samples to be used in loss computation
):
    
    preds = np.argmax(x.predict(x_dataset[sample], verbose = 0), axis = -1)
    return 1. - accuracy_score(preds,y_dataset[sample])


##########################################################################################
# Gradient Computations
##########################################################################################

def gradient(
    x = build_model(),
    sample = range(len(load_environment(config['env_name'])[0])),
    x_dataset = load_environment(config['env_name'])[0],
    y_dataset = load_environment(config['env_name'])[4],
):
    
    # Track loss computation
    with tf.GradientTape() as tape:
        loss = f(x,x_dataset,y_dataset,sample)
    
    # Return gradient
    return tape.gradient(loss, x.trainable_variables)

def smoothed_gradient(
    x = build_model(),
    x_pm_sigmau = build_model(), # time saver: copy of x, weights don't matter
    grad_approx = build_model(), # time saver: copy of x, weights don't matter
    sigmau = build_model(), # time saver: copy of x, weights don't matter
    sigma = config['s'],
    n = config['num_mc_fsigma'],
    sample = range(len(load_environment(config['env_name'])[0])),
    x_dataset = load_environment(config['env_name'])[0],
    y_dataset = load_environment(config['env_name'])[4],
):
    
    # Zero out gradient approximation, we'll be adding to it
    for i in grad_approx.trainable_weights:
        i.assign(tf.Variable(np.zeros(shape = i.numpy().shape), dtype='float32'))
    
    for _ in range(n):

        # create model with x as weights
        x_pm_sigmau.set_weights(x.get_weights())

        # create random model (i.e. \sigma u)
        for i in sigmau.trainable_weights:
            rand_var = tf.Variable(np.random.normal(
                size = i.numpy().shape,
                scale = sigma,
            ), dtype='float32')
            i.assign(rand_var)
        
        # add randomness to model (i.e. x+\sigma u)
        for i in range(len(x.trainable_weights)):
            x_pm_sigmau.trainable_weights[i].assign(tf.math.add(
                x_pm_sigmau.trainable_weights[i],
                sigmau.trainable_weights[i]
            ))
        
        # evaluate f(x+\sigma u)
        f_plus = f(x_pm_sigmau,x_dataset,y_dataset,sample).numpy()

        # subtract randomness to model (i.e. x-\sigma u)
        for i in range(len(x.trainable_weights)):
            x_pm_sigmau.trainable_weights[i].assign(tf.math.subtract(
                x_pm_sigmau.trainable_weights[i],
                tf.math.multiply(
                    sigmau.trainable_weights[i],
                    2
                )
            ))
        
        # evaluate f(x-\sigma u)
        f_minus = f(x_pm_sigmau,x_dataset,y_dataset,sample).numpy()

        # Compute gradient approximation (i.e. u(f(x+su)-f(x-su))/s)
        diff = (f_plus - f_minus) / (sigma * n)
        for i in sigmau.trainable_weights:
            i.assign(tf.math.multiply(i,diff))

        # Add approximation to list
        for i in range(len(grad_approx.trainable_weights)):
            grad_approx.trainable_weights[i].assign(tf.math.add(
                grad_approx.trainable_weights[i],
                sigmau.trainable_weights[i]
            ))
        
    # Return gradient approximation
    return grad_approx.trainable_weights


##########################################################################################
# Optimizers (each performs one optimization step)
##########################################################################################

def sgd(
    x = build_model(),
    x_iter = None,
    v_iter = None,
    x_pm_sigma = None,
    grad_approx = None,
    sigmau = None,
    lr = config['lr'],
    n_sgd = config['num_sgd'],
    n_inner_sgd = None,
    n_mc = None,
    m = None,
    sigma = None,
    x_dataset = load_environment(config['env_name'])[0],
    y_dataset = load_environment(config['env_name'])[4],
):
    
    sample = np.random.choice(range(len(x_dataset)),n_sgd)

    grad = gradient(x,sample,x_dataset,y_dataset)

    for i in range(len(x.trainable_weights)):
        x.trainable_weights[i].assign(tf.math.subtract(
            x.trainable_weights[i],
            tf.math.multiply(
                grad[i],
                lr
            )
        ))

    return x

def svrg(
    x = build_model(),
    x_iter = build_model(), # time saver: copy of x, weights don't matter
    v_iter = build_model(), # time saver: copy of x, weights don't matter
    x_pm_sigma = None,
    grad_approx = None,
    sigmau = None,
    lr = config['lr'],
    n_sgd = config['num_sgd'],
    n_inner_sgd = config['inner_svrg_num_sgd'],
    n_mc = None,
    m = config['inner_svrg_iterations'],
    sigma = None,
    x_dataset = load_environment(config['env_name'])[0],
    y_dataset = load_environment(config['env_name'])[4],
):
    
    # Full gradient descent
    mu = gradient(x,range(len(x_dataset)),x_dataset,y_dataset)

    # Create saved copy of model
    x_iter.set_weights(x.get_weights())

    # Option 1: Randomly select the gradient
    # inner_length = np.random.randint(1,m+1)

    # Option 2: Pick last gradient
    inner_length = m

    for _ in tqdm(range(inner_length), desc = 'SVRG inner'):
        
        sample = np.random.choice(range(len(x_dataset)),n_inner_sgd)
        iter_grad = gradient(x_iter,sample,x_dataset,y_dataset)
        tilde_grad = gradient(x,sample,x_dataset,y_dataset)

        for i in range(len(v_iter.trainable_weights)):
            v_iter.trainable_weights[i].assign(
                tf.math.add(
                    tf.math.subtract(
                        iter_grad[i],
                        tilde_grad[i]
                    ),
                    mu[i]
                )
            )
        
        for i in range(len(x_iter.trainable_weights)):
            x_iter.trainable_weights[i].assign(tf.math.subtract(
                x_iter.trainable_weights[i],
                tf.math.multiply(
                    v_iter.trainable_weights[i],
                    lr
                )
            ))
    
    x.set_weights(x_iter.get_weights())

    return x

def gssgd(
    x = build_model(),
    x_iter = None,
    v_iter = None,
    x_pm_sigmau = build_model(), # time saver: copy of x, weights don't matter
    grad_approx = build_model(), # time saver: copy of x, weights don't matter
    sigmau = build_model(), # time saver: copy of x, weights don't matter
    lr = config['lr'],
    n_sgd = config['num_sgd'],
    n_inner_sgd = None,
    n_mc = config['num_mc_fsigma'],
    m = None,
    sigma = config['s'],
    x_dataset = load_environment(config['env_name'])[0],
    y_dataset = load_environment(config['env_name'])[4],
):
    
    sample = np.random.choice(range(len(x_dataset)),n_sgd)

    grad = smoothed_gradient(x,x_pm_sigmau,grad_approx,sigmau,sigma,n_mc,sample,x_dataset,y_dataset)

    for i in range(len(x.trainable_weights)):
        x.trainable_weights[i].assign(tf.math.subtract(
            x.trainable_weights[i],
            tf.math.multiply(
                grad[i],
                lr
            )
        ))

    return x

def gssvrg(
    x = build_model(),
    x_iter = build_model(), # time saver: copy of x, weights don't matter
    v_iter = build_model(), # time saver: copy of x, weights don't matter
    x_pm_sigmau = build_model(), # time saver: copy of x, weights don't matter
    grad_approx = build_model(), # time saver: copy of x, weights don't matter
    sigmau = build_model(), # time saver: copy of x, weights don't matter
    lr = config['lr'],
    n_sgd = None,
    n_inner_sgd = config['inner_svrg_num_sgd'],
    n_mc = config['num_mc_fsigma'],
    m = config['inner_svrg_iterations'],
    sigma = config['s'],
    x_dataset = load_environment(config['env_name'])[0],
    y_dataset = load_environment(config['env_name'])[4],
):
    
    if tau == 0:
        mu = gradient(x,range(len(x_dataset)),x_dataset,y_dataset)
    else:
        mu = smoothed_gradient(x,x_pm_sigmau,grad_approx,sigmau,sigma,n_mc,range(len(x_dataset)),x_dataset,y_dataset)

    # Create saved copy of model
    x_iter.set_weights(x.get_weights())

    for _ in tqdm(range(m), desc = 'GSSVRG inner'):

        sample = np.random.choice(range(len(x_dataset)),n_inner_sgd)
        iter_grad = smoothed_gradient(x_iter,x_pm_sigmau,grad_approx,sigmau,sigma,n_mc,sample,x_dataset,y_dataset)
        if tau == 0:
            tilde_grad = gradient(x,sample,x_dataset,y_dataset)
        else:
            tilde_grad = smoothed_gradient(x,x_pm_sigmau,grad_approx,sigmau,sigma,n_mc,sample,x_dataset,y_dataset)

        for i in range(len(v_iter.trainable_weights)):
            v_iter.trainable_weights[i].assign(
                tf.math.add(
                    tf.math.subtract(
                        iter_grad[i],
                        tilde_grad[i]
                    ),
                    mu[i]
                )
            )
        
        for i in range(len(x_iter.trainable_weights)):
            x_iter.trainable_weights[i].assign(tf.math.subtract(
                x_iter.trainable_weights[i],
                tf.math.multiply(
                    v_iter.trainable_weights[i],
                    lr
                )
            ))
        
    x.set_weights(x_iter.get_weights())
    
    return x


##########################################################################################
# Logging
##########################################################################################
def log(
    filename = 'temp',
    step = -1,
    sigma = config['s'],
    x = build_model(),
    x_tr = load_environment(config['env_name'])[0],
    y_tr = load_environment(config['env_name'])[1],
    x_ts = load_environment(config['env_name'])[2],
    y_ts = load_environment(config['env_name'])[3],
    y_tr_sparse = load_environment(config['env_name'])[4],
    y_ts_sparse = load_environment(config['env_name'])[5],
    dataframe = pd.DataFrame(
        data = [np.zeros(6)],
        columns = [
            'step',
            'sigma',
            'train_loss',
            'train_misclassification_rate',
            'test_loss',
            'test_misclassification_rate'
        ]
    ),
    first = True
):
    
    dataframe.iloc[0] = [
        step,
        sigma,
        f(x,x_tr,y_tr_sparse,range(len(x_tr))).numpy(),
        misclassification_rate(x,x_tr,y_tr,range(len(x_tr))),
        f(x,x_ts,y_ts_sparse,range(len(x_ts))).numpy(),
        misclassification_rate(x,x_ts,y_ts,range(len(x_ts))),
    ]

    if first:
        dataframe.to_csv(f'./logs/{filename}.csv', index=False)
    else:
        dataframe.to_csv(f'./logs/{filename}.csv', mode='a', index=False, header=False)
    



##########################################################################################
# Run Optimization Algorithm
##########################################################################################
def optimize(
    steps = config['steps'],
    save_freq = config['save_freq'], # how many steps between logging
    name = config['name'],
    env_name = config['env_name'],
    optimizer_name = config['optimizer'][0],
    x = build_model(),
    x_iter = build_model(), # time saver: copy of x, weights don't matter
    v_iter = build_model(), # time saver: copy of x, weights don't matter
    x_pm_sigmau = build_model(), # time saver: copy of x, weights don't matter
    grad_approx = build_model(), # time saver: copy of x, weights don't matter
    sigmau = build_model(), # time saver: copy of x, weights don't matter
    lr = config['lr'],
    n_sgd = config['num_sgd'],
    n_inner_sgd = config['inner_svrg_num_sgd'],
    n_mc = config['num_mc_fsigma'],
    m = config['inner_svrg_iterations'],
    sigma = config['s'],
):
    
    initialization = 0
    np.random.seed(config['seed'])

    # Load Data
    x_train,y_train,x_test,y_test,y_train_sparse,y_test_sparse = load_environment(env_name = env_name)

    # Pick optimizer
    if optimizer_name == 'sgd':
        optimizer = sgd
    elif optimizer_name == 'svrg':
        optimizer = svrg
    elif optimizer_name == 'gssgd':
        optimizer = gssgd
    elif optimizer_name == 'gssvrg':
        optimizer = gssvrg
    else:
        print(f'Optimizer "{optimizer_name}" not coded.')
    
    # Initialize logging
    logging_df = pd.DataFrame(
        data = [np.zeros(6)],
        columns = [
            'step',
            'sigma',
            'train_loss',
            'train_misclassification_rate',
            'test_loss',
            'test_misclassification_rate'
        ]
    )
    
    log(
        filename = f'{name}_{env_name}_{optimizer_name}',
        step = 0,
        sigma = sigma,
        x = x,
        x_tr = x_train,
        y_tr = y_train,
        x_ts = x_test,
        y_ts = y_test,
        y_tr_sparse = y_train_sparse,
        y_ts_sparse = y_test_sparse,
        dataframe = logging_df,
        first = True
    )

    for counter in tqdm(range(steps)):

        x = optimizer(x,x_iter,v_iter,x_pm_sigmau,grad_approx,sigmau,lr,n_sgd,n_inner_sgd,n_mc,m,sigma,x_train,y_train_sparse)

        if (counter + 1) % save_freq == 0:

            log(
                filename = f'{name}_{env_name}_{optimizer_name}',
                step = counter + 1,
                sigma = sigma,
                x = x,
                x_tr = x_train,
                y_tr = y_train,
                x_ts = x_test,
                y_ts = y_test,
                y_tr_sparse = y_train_sparse,
                y_ts_sparse = y_test_sparse,
                dataframe = logging_df,
                first = False
            )

            # if (optimizer_name == 'gssgd') and (sigma > 0.001):
            # if sigma > 0.001:
            #     if initialization == 1:
            #         sigma = 0.5 * sigma
            #         initialization = 0
            #     else:
            #         initialization += 1

if __name__ == '__main__':

    for alg in config['optimizer']:
        learning_rate = config[f'{alg}_lr']
        if alg == 'gssvrg_tau':
            name = f"{config['name']}_tau"
            alg_name = 'gssvrg'
            tau = 0
        else:
            name = config['name']
            alg_name = alg
            tau = 1
        print(alg.upper())
        optimize(
            steps = config['steps'],
            save_freq = config['save_freq'], # how many steps between logging
            name = name,
            env_name = config['env_name'],
            optimizer_name = alg_name,
            x = build_model(),
            x_iter = build_model(), # time saver: copy of x, weights don't matter
            v_iter = build_model(), # time saver: copy of x, weights don't matter
            x_pm_sigmau = build_model(), # time saver: copy of x, weights don't matter
            grad_approx = build_model(), # time saver: copy of x, weights don't matter
            sigmau = build_model(), # time saver: copy of x, weights don't matter
            lr = config[f'{alg}_lr'],
            n_sgd = config['num_sgd'],
            n_inner_sgd = config['inner_svrg_num_sgd'],
            n_mc = config['num_mc_fsigma'],
            m = config['inner_svrg_iterations'],
            sigma = config['s'],
        )