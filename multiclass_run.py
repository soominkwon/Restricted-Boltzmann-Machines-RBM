#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 14:24:30 2020

@author: soominkwon
"""

from multiclass_rbm import RBM
import tensorflow as tf
import argparse
import imageio
import kerastuner as kt

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='RBM')

parser.add_argument('--checkpoint_path', type=str, default="logdir/check_{epoch}/cp-{epoch:02d}.ckpt", 
                    help='path to save model')
parser.add_argument('--save', type=bool, default=True, help='saves model and checkpoints')
parser.add_argument('--load', type=bool, default=False, help='loads model and checkpoints')
parser.add_argument('--dist_type_vis', type=str, default="bernoulli", help='visible distribution type.')
parser.add_argument('--dist_type_hid', type=str, default="bernoulli", help='hidden distribution type.')
parser.add_argument('--batch_size', type=int, default=200, help='batch size.')
parser.add_argument('--n_vis', type=int, default=784, help='number of visible units.')
parser.add_argument('--n_hid', type=int, default=200, help='number of hidden units.')
parser.add_argument('--cd_k', type=int, default=1, help='number of contrastive divergence iterations')
parser.add_argument('--epoch', type=int, default=5, help='number of training loops')
parser.add_argument('--v_marg_steps', type=int, default=250, help='number of training loops')
parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
parser.add_argument('--use_tuner', type=int, default=False, help='use Keras hyperparameter tuning')
#parser.add_argument('--optimizer', type=str, default="SGD", help='choose to use Adam or SGD optimizer')

args = parser.parse_args()

# loading MNIST dataset
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
bool_mat = x_train > 0
x_train = (bool_mat*1).astype("float32")

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)


# define model for parameter tuning
def model_builder(hp):    
    hp_hid = hp.Int('n_hidden', min_value=100, max_value=700, step=100)
    args.n_hid = hp_hid
    
    rbm = RBM(args=args)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    rbm.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp_learning_rate))

    return rbm

# defining main function
def main():
    rbm = RBM(args=args)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    rbm.compile(optimizer=optimizer)
    
    if args.save:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_freq='epoch')
        rbm.fit(train_dataset, epochs=args.epoch, callbacks=[cp_callback])
        rbm.Functions.sample_v_marg()
    
    else:
        rbm.fit(train_dataset, epochs=args.epoch)
        rbm.Functions.sample_v_marg()
        

if __name__ == "__main__":
    """ Working example of using the tuner + training the RBM model on the MNIST dataset.
        With this current build, we can use the keras hypertuner to determine the learning rate,
        type of optimizer (e.g. ADAM, SGD), and the batch size. 
    """
    
    if args.use_tuner:
        tuner = kt.Hyperband(model_builder,
                      objective=kt.Objective('PLL Loss', 'min'),
                      max_epochs=5,
                      directory='tuner_results',
                      project_name='mnist_trial')

        tuner.search(train_dataset, epochs=args.epoch)
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        print(best_hps.get('n_hidden'))
        
    else:
        main()
    