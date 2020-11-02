#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 12:58:50 2020

@author: soominkwon
"""

import tensorflow as tf
from tensorflow import keras
from dist_util import sample_bernoulli
import numpy as np
import os
from util import save_merged_images
import imageio
import kerastuner as kt

class Positive(keras.layers.Layer):
    """ Defines the positive phase for our contrastive divergence function. Used for sampling
        p(h|v) and its activations.
        
        Arguments:
            args: parse_args input command-line arguments (hyperparameters)
            W:    Weight matrix defined in model class
            b:    Hidden bias vector defined in model class
    """
        
    def __init__(self, args, W, b):
        super(Positive, self).__init__()
        
        self.W = W
        self.b = b
        
        if args.dist_type_hid == "bernoulli":
            self.presample_h_distribution = tf.nn.sigmoid
            self.sample_h_distribution = sample_bernoulli
            
    def call(self, inputs):
        prob_h_given_v = self.presample_h_distribution(tf.matmul(inputs, self.W) + self.b)
        hid_activations = self.sample_h_distribution(prob_h_given_v)
        
        return prob_h_given_v, hid_activations
        
        
class Negative(keras.layers.Layer):
    """ Defines the positive phase for our contrastive divergence function. Used for sampling
        p(v|h) and its activations.
        
        Arguments:
            args: parse_args input command-line arguments (hyperparameters)
            W:    Weight matrix defined in model class
            a:    Visible bias vector defined in model class            
    """
    
    def __init__(self, args, W, a):
        super(Negative, self).__init__()
        
        self.W = W
        self.a = a
        
        if args.dist_type_vis == "bernoulli":
            self.presample_v_distribution = tf.nn.sigmoid
            self.sample_v_distribution = sample_bernoulli
            
    def call(self, inputs):
        prob_v_given_h = self.presample_v_distribution(tf.matmul(inputs, tf.transpose(self.W)) + self.a)
        vis_activations = self.sample_v_distribution(prob_v_given_h)
        
        return prob_v_given_h, vis_activations


class Functions:
    """ This class defines the functionalities needed for the RBM. These functionalities include
        contrastive_divergence(), free_energy(), pseudo_log_likelihood(), and gibbs_sampling().
        
        Arguments:
            args: parse_args input command-line arguments (hyperparameters)
            W:    Weight matrix defined in model class
            a:    Visible bias vector defined in model class
            b:    Hidden bias vector defined in model class
    """
    
    def __init__(self, args, W, a, b):
        super(Functions, self).__init__()
        
        self.args = args
        self.n_visible = self.args.n_vis
        
        self.W = W
        self.a = a
        self.b = b
        
        self._idx_pll = 0 # index used for PLL calculation
        
        self.Positive = Positive(args=self.args, W=self.W, b=self.b)
        self.Negative = Negative(args=self.args, W=self.W, a=self.a)
        
    def contrastive_divergence(self, inputs):
        """ Defines the core operations that will be used for training. RBM model trains
            and computes gradients via contrastive divergence.
            
            Arguments:
                inputs: Data that will be used to compute gradients
            
            Returns:
                grads: Gradients for computation
        
        """
        
        # positive phase begins        
        positive_hidden_probs, positive_hidden_activations = self.Positive(inputs)
        positive_grads = tf.matmul(tf.transpose(inputs), positive_hidden_probs)
        
        # negative phase begins
        hidden_activations = positive_hidden_activations
        
        # contrastive divergence iterations
        for step in range(self.args.cd_k):
            visible_probs, visible_activations = self.Negative(hidden_activations)
            hidden_probs, hidden_activations = self.Positive(visible_activations)
            
        negative_visible_activations = visible_activations
        negative_hidden_activations = hidden_activations
            
        negative_grads = tf.matmul(tf.transpose(negative_visible_activations), negative_hidden_activations)
            
        # calculate gradients
        grad_w_new = -(positive_grads - negative_grads) / tf.compat.v1.to_float(tf.shape(inputs)[0])
        grad_visible_bias_new = -(tf.reduce_mean(inputs - negative_visible_activations, 0))
        grad_hidden_bias_new = -(tf.reduce_mean(positive_hidden_probs - negative_hidden_activations, 0))
            
        grads = [grad_w_new, grad_hidden_bias_new, grad_visible_bias_new]
            
        return grads
    
    def free_energy(self, inputs):
        """ Computes the free energy of the visible input.
        
            Arguments:
                inputs: Input for free_energy calculation
            
            Returns:
                fe: Free energy
        """
        
        fe = tf.reduce_mean(- tf.matmul(inputs, tf.expand_dims(self.a, -1)) \
            - tf.reduce_sum(tf.math.log(1 + tf.exp(self.b + tf.matmul(inputs, self.W))), axis=1), axis=0)

        return fe
    
    def pseudo_log_likelihood(self, inputs):
        """ Computes PLL.
        
            Arguments:
                inputs: Input to compute pseudo log likelihood
            
        """
        
        v = inputs
        vi = tf.concat(
            [v[:, :self._idx_pll + 1], 1 - v[:, self._idx_pll + 1:self._idx_pll + 2], v[:, self._idx_pll + 2:]], 1)
        self._idx_pll = (self._idx_pll + 1) % self.n_visible
        fe_x = self.free_energy(v)
        fe_xi = self.free_energy(vi)
        return tf.reduce_mean(tf.reduce_mean(
            self.n_visible * tf.math.log(tf.nn.sigmoid(tf.clip_by_value(fe_xi - fe_x, -20, 20))), axis=0))

    def gibbs_sample(self, inputs, steps=1):
        """ Perform n-steps Gibbs sampling chain to inspect the marginal distribution between
            the input and visible variables.
            
        """
        
        v = inputs
        
        for step in range(steps):
            h_given_v, h = self.Positive(v)
            v_given_h, v = self.Negative(h)
        
        return v
            
    def sample_v_marg(self, size=784, n_samples=100):
        """ This function samples images via Gibbs sampling to inspect the marginal distribution
        of the visible variables.
        
        """
        
        batch_v_noise = np.random.rand(n_samples, size)
        v_marg = self.gibbs_sample(batch_v_noise, steps=self.args.v_marg_steps)
        #v_marg = v_marg.eval(session=tf.compat.v1.Session())
        v_marg = np.reshape(v_marg, (n_samples, 28, 28))
        save_merged_images(images=v_marg, size=(10, 10), path=os.path.join(*[
            'run_' + '_steps' + str(self.args.v_marg_steps) + '.png']))
                                
    
class RBM(keras.Model): 
    """ Combines the Positive pass and Negative pass classes into an end-to-end model for
        training. The loss function we will be using will the pseudo log likelihood computation
        between inputs and variables.
        
        Arguments:
            args: parse_args input command-line arguments (hyperparameters)
    """
        
    def __init__(self, args):
        super(RBM, self).__init__()
        
        self.args = args
        
        self.n_visible = self.args.n_vis
        self.n_hidden = self.args.n_hid
        # self.n_hidden = n_hid
        
        # creating parameters
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.W = tf.Variable(initial_value=w_init(shape=(self.n_visible, self.n_hidden), dtype=tf.float32),
                             trainable=True)
        
        a_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        self.a = tf.Variable(initial_value=a_init(shape=(self.n_visible,), dtype=tf.float32),
                            trainable=True)
        
        b_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        self.b = tf.Variable(initial_value=b_init(shape=(self.n_hidden,), dtype=tf.float32),
                             trainable=True)
    
        # calling subclasses
        self.Positive = Positive(args=self.args, W=self.W, b=self.b)
        self.Negative = Negative(args=self.args, W=self.W, a=self.a)
        self.Functions = Functions(args=self.args, W=self.W, a=self.a, b=self.b)
        
    def compile(self, optimizer):
        super(RBM, self).compile()

        self.optimizer = optimizer

    def train_step(self, inputs):
        print(inputs)
        print(inputs.shape)
        grads = self.Functions.contrastive_divergence(inputs=inputs)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        pll_loss = self.Functions.pseudo_log_likelihood(inputs=inputs)
        # free_energy = self.Functions.free_energy(inputs=inputs)
        # self.Functions.sample_v_marg()
        
        return {"PLL Loss": pll_loss}

    def call(self, inputs):
        """ This function includes the free energy computation as the loss function as well as calls
            the other classes for the positive and negative phase of training.
            
            Arguments:
                inputs: visible inputs for free energy computation
        """
        
        #pos_hidden_probs, pos_hidden_acts = self.Positive(inputs=inputs)
        #visible_probs, visible_activations = self.Negative(inputs=pos_hidden_acts)
        visible_activations = self.gibbs_sample(inputs=inputs)
        # add free energy as loss function
        pll_loss = self.Functions.pseudo_log_likelihood(inputs=inputs)
        
        #fe_loss = self.Functions.free_energy(inputs=inputs)
        self.add_loss(pll_loss)
        
        return visible_activations        


#if __name__ == "__main__":
   