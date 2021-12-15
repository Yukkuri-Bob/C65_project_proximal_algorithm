#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:19:56 2021

@edited for C65 project 2021

Modified version for usage of report. This version should work on tensorflow 2.*.*.

model1 provides an [Inside]*800*[output] fully connected NN.

model2 provides an [InSize]*4000*1000*200*25*[OutSize] fully connected NN, using Leaky ReLU instead.

One can change the fc part to add/delete/edit layers.

The SVRG part has been annoted as this is beyond the topic of report.

Original introduction is given below.
"""

"""!@package models

Several feedforawrd neural network models used in the paper.

We implement two fully connected models with size [InSize]x100x[OutSize] and [InSize]x800x[OutSize] where [InSize] and [OutSize] are input and output dimensions.

Copyright (c) 2019 Nhan H. Pham, Department of Statistics and Operations Research, University of North Carolina at Chapel Hill

Copyright (c) 2019 Quoc Tran-Dinh, Department of Statistics and Operations Research, University of North Carolina at Chapel Hill

Copyright (c) 2019 Lam M. Nguyen, IBM Research, Thomas J. Watson Research Center
Yorktown Heights

Copyright (c) 2019 Dzung T. Phan, IBM Research, Thomas J. Watson Research Center
Yorktown Heights
All rights reserved.

If you found this helpful and are using it within our software please cite the following publication:

* N. H. Pham, L. M. Nguyen, D. T. Phan, and Q. Tran-Dinh, **[ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization](https://arxiv.org/abs/1902.05679)**, _arXiv preprint arXiv:1902.05679_, 2019.

"""

import tensorflow.compat.v1 as tf
from tf_slim.layers import flatten
from utils import *

tf.disable_v2_behavior()

#==================================================================================================================
# ================ Tensorflow Model =======================

def model1(x, input_size, output_size, seed=114514):
    
    """! Fully connected model [InSize]*800*[OutSize]

	Implementation of a [InSize]*800*[OutSize] fully connected model.

	Parameters
	----------
	@param x : placeholder for input data
	@param input_size : size of input data
	@param output_size : size of output data
	    
	Returns
	-------
	@retval logits : output
	@retval logits_dup : a copy of output
	@retval w_list : trainable parameters
	@retval w_list_dup : a copy of trainable parameters
	"""

	#==================================================================================================================
	## model definition
    mu = 0
    sigma = 0.2
    weights = {
	    'wfc1': tf.Variable(tf.truncated_normal(shape=(input_size,800), mean = mu, stddev = sigma, seed = seed)),
	    'out': tf.Variable(tf.truncated_normal(shape=(800,output_size), mean = mu, stddev = sigma, seed = seed))
	}

    biases = {
	    'bfc1': tf.Variable(tf.zeros(800)),
	    'out': tf.Variable(tf.zeros(output_size))
	}

	# Flatten input.
    c_flat = flatten(x)

	# Layers  
	# Activation is ReLU
    fc1 = fc_relu(c_flat, weights['wfc1'], biases['bfc1'])
    
	# Out
    logits = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    w_list = []
    for w,b in zip(weights, biases):
	    w_list.append(weights[w])
	    w_list.append(biases[b])
        
    return logits, w_list,


	#==================================================================================================================
	## duplicate the model used in ProxSVRG
	#weights_dup = {
	#    'wfc1': tf.Variable(tf.truncated_normal(shape=(input_size,1000), mean = mu, stddev = sigma, seed = seed)),
    #    'wfc2': tf.Variable(tf.truncated_normal(shape=(1000,200), mean = mu, stddev = sigma, seed = seed)),
    #    'wfc3': tf.Variable(tf.truncated_normal(shape=(200,50), mean = mu, stddev = sigma, seed = seed)),
	#    'out': tf.Variable(tf.truncated_normal(shape=(50,output_size), mean = mu, stddev = sigma, seed = seed))
	#}

	#biases_dup = {
	#    'bfc1': tf.Variable(tf.zeros(1000)),
    #    'bfc2': tf.Variable(tf.zeros(200)),
    #    'bfc3': tf.Variable(tf.zeros(50)),
	#    'out': tf.Variable(tf.zeros(output_size))
	#}

	# Flatten input.
	#c_flat_dup = flatten(x)

	# Layer 1: Fully Connected. Input = input_size. Output = 800.    
	# Activation.
	#fc1_dup = fc_relu(c_flat_dup, weights_dup['wfc1'], biases_dup['bfc1'])
	#fc2_dup = fc_relu(fc1_dup, weights_dup['wfc2'], biases_dup['bfc2'])
	#fc3_dup = fc_relu(fc2_dup, weights_dup['wfc3'], biases_dup['bfc3'])
  

	# Layer 2: Fully Connected. Input = 800. Output = output_size.
	#logits_dup = tf.add(tf.matmul(fc3_dup, weights_dup['out']), biases_dup['out'])

	#w_list_dup = []
	#for w,b in zip(weights_dup, biases_dup):
	#    w_list_dup.append(weights_dup[w])
	#    w_list_dup.append(biases_dup[b])

	#return logits, logits_dup, w_list, w_list_dupreturn logits, w_list,
    
def model2(x, input_size, output_size, seed=114514):
    
    """! Fully connected model [InSize]*4000*1000*200*25*[OutSize]

	Implementation of a [InSize]*4000*1000*200*25*[OutSize] fully connected model.

	Parameters
	----------
	@param x : placeholder for input data
	@param input_size : size of input data
	@param output_size : size of output data
	    
	Returns
	-------
	@retval logits : output
	@retval logits_dup : a copy of output
	@retval w_list : trainable parameters
	@retval w_list_dup : a copy of trainable parameters
	"""

	#==================================================================================================================
	## model definition
    mu = 0
    sigma = 0.15
    weights = {
	    'wfc1': tf.Variable(tf.truncated_normal(shape=(input_size,4000), mean = mu, stddev = sigma, seed = seed)),
        'wfc2': tf.Variable(tf.truncated_normal(shape=(4000,1000), mean = mu, stddev = sigma, seed = seed)),
        'wfc3': tf.Variable(tf.truncated_normal(shape=(1000,200), mean = mu, stddev = sigma, seed = seed)),
        'wfc4': tf.Variable(tf.truncated_normal(shape=(200,25), mean = mu, stddev = sigma, seed = seed)),
	    'out': tf.Variable(tf.truncated_normal(shape=(25,output_size), mean = mu, stddev = sigma, seed = seed))
	}

    biases = {
	    'bfc1': tf.Variable(tf.zeros(4000)),
        'bfc2': tf.Variable(tf.zeros(1000)),
        'bfc3': tf.Variable(tf.zeros(200)),
        'bfc4': tf.Variable(tf.zeros(25)),
	    'out': tf.Variable(tf.zeros(output_size))
	}

	# Flatten input.
    c_flat = flatten(x)

	# Layers  
	# Activation is ReLU
    fc1 = fc_leaky_relu(c_flat, weights['wfc1'], biases['bfc1'])
    fc2 = fc_leaky_relu(fc1, weights['wfc2'], biases['bfc2'])
    fc3 = fc_leaky_relu(fc2, weights['wfc3'], biases['bfc3'])
    fc4 = fc_leaky_relu(fc3, weights['wfc4'], biases['bfc4'])
    
	# Out
    logits = tf.add(tf.matmul(fc4, weights['out']), biases['out'])

    w_list = []
    for w,b in zip(weights, biases):
	    w_list.append(weights[w])
	    w_list.append(biases[b])
        
    return logits, w_list,
    