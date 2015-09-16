# -*- coding: utf-8 -*-

import numpy as np 

import theano
import theano.tensor as T

from utils import *

class LookupLayer(object):
    """
    Classic Lookup Layer taking as inputs the length of word vocabulary 'l_vocab_w' and the size of word embeddings 'n_f'.
    The Lookup matrix 'WE' is the parameter and its gradient is computed with respect to 'output' to avoid computing updates
    for the whole matrix.
    """
    def __init__(self, word_id, l_vocab_w, n_f):
	
	self.WE = globals()['init_zeros']((l_vocab_w, n_f), 'WE')
        self.output = self.WE[word_id]
        self.params = [(self.WE,self.output)]


class EmbLayer(object):
    """
    Layer used to compute word embeddings from their characters, using a convolution filter.
    Inputs:
    word_* : Theano tensors that will be recipients for the inputs. Each will contain the informations for a whole sentence.
    word_emb : matrix of integers, each row containing the indexes of the characters (in the character vocabulary) forming a character n-gram. With padding, it will contain as many rows as there is characters in the sentence.
    word_len : vector of integers, each one indicating the number of characters in each word. Containing as many integers as there is words in the sentence.
    word_beg : vector of integers, each one indicating the position of the character that begins the word in the sentence. ( Information redundant with the previous tensor, but there for comfort). Containing as many integers as there is words in the sentence.
    word_id : vector of integers, each one containing the index of a word (in the word vocabulary). Containing as many integers as there is words in the sentence.
    l_vocab_w : length of the word vocabulary. If we don't use it, to 'None'
    l_vocab : length of the character vocabulary
    n_char : dimension of the character embeddings
    winDim : size of the character n-grams - could be deduced from data but there for comfort
    n_f : dimension of the word embeddings
    init : to choose from 'Initialization functions', will initiate the transition matrices and initial hidden states for the bi-RNN (word and character lookup matrices are initialized to zero)
    """
    def __init__(self, word_emb, word_len, word_beg, word_id, l_vocab_w, l_vocab, n_char, winDim, n_f, init):
	
	self.C = globals()['init_zeros']((l_vocab, n_char), 'C')
	self.F = globals()[init]((n_char * winDim, n_f ), 'F')
	self.params = [(self.C, self.C), (self.F, self.F)]
        self.emb = word_emb

	""" 
        Lambda function to obtain the concatenation of the character embeddings that represent the character n-gram, and then apply the convolution.
        """
	get_n_gram_E = lambda input : T.dot(self.C[input].reshape((-1,n_char*winDim)), self.F)
        
        if l_vocab_w != None:
        
            self.WE = globals()['init_zeros']((l_vocab_w, n_f), 'WE')
            self.WE_ind = self.WE[word_id]
            self.params.append((self.WE,self.WE_ind))
        

	    """
	    One scan - that process a word: getting the result of the n-gram representations from input after convolution. The function then do a max pooling and concatenates the word embedding to it.
	    """
            def get_WE(len, beg, we):
                res, seq = theano.scan( fn=get_n_gram_E, \
                                            sequences = self.emb[beg:beg+len] )
                return T.concatenate([T.tanh(res.max(axis=0, keepdims=False)).reshape((n_f,)),we])
         
            def get_SE(seq_len, seq_beg, seq_we):
                res, seq = theano.scan(fn=get_WE, \
                                   sequences = [seq_len, seq_beg, seq_we] )
            
                return res.reshape((seq_len.shape[0],n_f*2))   

            self.output = get_SE(word_len, word_beg, self.WE_ind)

        else:
            def get_WE(len, beg):
                res, seq = theano.scan( fn=get_n_gram_E, \
                                            sequences = self.emb[beg:beg+len] )
		return T.tanh(res.max(axis=0, keepdims=False)).reshape((n_f,))

            def get_SE(seq_len, seq_beg):
                res, seq = theano.scan(fn=get_WE, \
                                           sequences = [seq_len, seq_beg] )
		return res.reshape((seq_len.shape[0],n_f))

	    self.output = get_SE(word_len, word_beg)
       

class EmbLayer_RNN(object):
    """
    Layer used to compute word embeddings from their characters, using RNNs.
    Inputs:
    word_* : Theano tensors that will be recipients for the inputs. Each will contain the informations for a whole sentence.
    word_emb : matrix of integers, each row containing the indexes of the characters (in the character vocabulary) forming a character n-gram. With padding, it will contain as many rows as there is characters in the sentence.
    word_len : vector of integers, each one indicating the number of characters in each word. Containing as many integers as there is words in the sentence.
    word_beg : vector of integers, each one indicating the position of the character that begins the word in the sentence. ( Information redundant with the previous tensor, but there for comfort). Containing as many integers as there is words in the sentence.
    word_id : vector of integers, each one containing the index of a word (in the word vocabulary). Containing as many integers as there is words in the sentence.
    l_vocab_w : length of the word vocabulary. If we don't use it, to 'None'
    l_vocab : length of the character vocabulary
    n_char : dimension of the character embeddings
    winDim : size of the character n-grams - could be deduced from data but there for comfort
    n_f : dimension of the word embeddings
    init : to choose from 'Initialization functions', will initiate the transition matrices and initial hidden states for the bi-RNN (word and character lookup matrices are initialized to zero)
    """
    def __init__(self, word_emb, word_len, word_beg, word_id, l_vocab_w, l_vocab, n_char, winDim, n_f, init):

	self.C = globals()['init_zeros']((l_vocab, n_char), 'C')
	self.W_c = globals()[init]((n_char * winDim, n_f / 2 ), 'W_c')
	self.W_t = globals()[init]((n_f / 2, n_f / 2 ), 'W_t')
	self.W_tr = globals()[init]((n_f / 2, n_f / 2 ), 'W_tr')
	self.h_0 = globals()[init]((n_f / 2,), 'h_0')
	self.h_n = globals()[init]((n_f / 2,), 'h_n')
	self.params = [(self.C, self.C), (self.W_c, self.W_c), (self.W_t, self.W_t), (self.W_tr, self.W_tr), (self.h_0, self.h_0), (self.h_n, self.h_n)]
	self.emb = word_emb
	
	""" 
        Lambda function to obtain the concatenation of the character embeddings that represent the character n-gram
        """
	get_n_gram_rep = lambda input : self.C[input].flatten()

	""" 
        Condition on the size of the word vocabulary will determine if we use a word embedding from a lookup matrix with the embedding we obtain from characters
        """
        if l_vocab_w != None:
        
	    """
	    Creates a lookup matrix for words, that is added as parameter, and its gradient will be computed wrt the subtensor with the words that are in the sentence.
 	    """ 
            self.WE = globals()['init_zeros']((l_vocab_w, n_f), 'WE')
            self.WE_ind = self.WE[word_id]
            self.params.append((self.WE,self.WE_ind))

	    """
	    Forward recursion
	    """
	    def rec_hidden(n_t, h_t1):
	        h_t = T.tanh(T.dot(n_t, self.W_c) + T.dot(h_t1, self.W_t))
	        return h_t
	    """
	    Backward recursion
	    """
   	    def rec_hidden_r(n_t, h_t1):
	        h_t = T.tanh(T.dot(n_t, self.W_c) + T.dot(h_t1, self.W_tr))
	        return h_t
	
	    """
	    3 parallel Scans - that process a word: getting the n-gram representations from input + lookup, forward, backward recursion. Returns concatenation of both final hidden states + word embedding from the lookup matrix.
	    """
            def get_W_E(len, beg, we):
                res, _ = theano.scan( fn=get_n_gram_rep, \
                                            sequences = self.emb[beg:beg+len] )
		rep, _ = theano.scan( fn = rec_hidden, \
					    sequences = res, \
					    outputs_info = self.h_0 )
		rep_r, _ = theano.scan( fn = rec_hidden_r, \
					    sequences = res[:,::-1], \
					    outputs_info = self.h_n )

		return T.concatenate([rep[-1],rep_r[-1],we])
	
	    """
	    Get the right inputs for each word and applies the previous function by a scan on words. Returns the embeddings for the whole sentence.
	    """
	    def get_SE(seq_len, seq_beg, seq_we):
                res, _ = theano.scan(fn=get_W_E, \
                                           sequences = [seq_len, seq_beg, seq_we] )
		return res.reshape((seq_len.shape[0],n_f * 2))

	    self.output = get_SE(word_len, word_beg, self.WE_ind)

        else:
	    def rec_hidden(n_t, h_t1):
	        h_t = T.tanh(T.dot(n_t, self.W_c) + T.dot(h_t1, self.W_t))
	        return h_t
	    def rec_hidden_r(n_t, h_t1):
	        h_t = T.tanh(T.dot(n_t, self.W_c) + T.dot(h_t1, self.W_tr))
	        return h_t
	
            def get_W_E(len, beg):
                res, _ = theano.scan( fn=get_n_gram_rep, \
                                            sequences = self.emb[beg:beg+len] )
		rep, _ = theano.scan( fn = rec_hidden, \
					    sequences = res, \
					    outputs_info = self.h_0 )
		rep_r, _ = theano.scan( fn = rec_hidden_r, \
					    sequences = res[:,::-1], \
					    outputs_info = self.h_n )
		return T.concatenate([rep[-1],rep_r[-1]])
	
	    def get_SE(seq_len, seq_beg):
                res, _ = theano.scan(fn=get_W_E, \
                                           sequences = [seq_len, seq_beg] )
		return res.reshape((seq_len.shape[0],n_f))

	    self.output = get_SE(word_len, word_beg)

	
