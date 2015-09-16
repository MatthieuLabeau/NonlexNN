# -*- coding: utf-8 -*-

import numpy as np 

import theano
import theano.tensor as T

# ---  Utilities : Initialization functions --- 
def init_ones(size, name):
    return theano.shared(value=np.ones(size, dtype='float64'), name=name, borrow=True)

def init_zeros(size, name):
    return theano.shared(value=np.zeros(size, dtype='float64'), name=name, borrow=True)

def init_uniform(size, name):
    return theano.shared(value=np.asarray(np.random.uniform(low = -np.sqrt(6. / np.sum(size)),
							high = np.sqrt(6. / np.sum(size)),
							size=size), dtype='float64'), name=name, borrow=True)

def init_ortho(size, name):
    W = np.random.randn(max(size[0],size[1]),min(size[0],size[1]))
    u, s, v = np.linalg.svd(W)
    return theano.shared(value=u.astype('float64')[:,:size[1]], name=name, borrow=True)

# --- Utilities : Activation functions ---

def relu(x):
    return T.switch(x<0, 0, x)

# --- Viterbi ---
"""
Function used to backtrack the best sequence using the max/argmax matrix comptued with Viterbi inside of the network. (It's here just because it's easier to write in python than theano)
"""
def backtrack(first, argmax):
    tokens = [first]
    for i in xrange(argmax.shape[0]-1, -1, -1):
        tokens.append(argmax[i, tokens[-1]])
    return tokens[::-1]       
