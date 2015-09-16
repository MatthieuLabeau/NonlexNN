
# -*- coding: utf-8 -*-

import numpy as np 

import theano
import theano.tensor as T

from utils import *
from emb import *
from hidden import *
from output import *
from trainer import *

"""
Class wrapping the whole model. 
Inputs: 
l_vocab_w - length of the word vocabulary - 'None' if we don't use one
l_vocab - length of the character vocabulary
l_vocab_out - length of the tag vocabulary
winDim - size of the character n-grams
n_char - dimension of the character embeddings
n_f - dimension of the word embeddings
n_hidden - dimension of the hidden embeddings 
traner - name of the trainer to use : 'AdagradTrainer', AdadeltaTrainer', or 'GDTrainer'
activation - name of the activation function to use, usually T.tanh or 'relu'
model - 'ff' is feedforward, 'biRNN' for bidirectional recurrent hidden Layer
viterbi - True or False for outputing a structured sequence using Viterbi or not
char_model - 'conv' for using convolution for the embedding layer, 'RNN' for using a recurrent layer
only_lexicalized - True or False. If true, will use only word embeddings (and no embeddings from characters).
Inputs are mostly choices for the architecture and the training that the wrapper class will apply. Once everything is set, theano functions are created to train and evaluate the model, and access the parameters.
"""
class Conv_model(object):	
    def __init__(self, l_vocab_w, l_vocab, l_vocab_out, winDim, n_char, n_f, n_hidden, lr, trainer = 'AdagradTrainer', activation=T.tanh, model = 'ff', viterbi = False, char_model = 'conv', only_lexicalized = False ):
        	
        word_emb = T.imatrix('word_emb')
        word_len = T.ivector('word_len')
        word_beg = T.ivector('word_beg')
	word_id = T.ivector('word_id')
        tags = T.ivector('tags')

	if only_lexicalized:
	    self.embLayer = LookupLayer(word_id, l_vocab_w = l_vocab_w, n_f=n_f)
	else:
	    if char_model == 'conv':
                self.embLayer = EmbLayer(word_emb, word_len, word_beg, word_id, l_vocab_w = l_vocab_w, l_vocab=l_vocab, n_char=n_char, winDim=winDim, n_f=n_f, init = 'init_uniform')
	    else:
	        self.embLayer = EmbLayer_RNN(word_emb, word_len, word_beg, word_id, l_vocab_w = l_vocab_w, l_vocab=l_vocab, n_char=n_char, winDim=winDim, n_f=n_f, init = 'init_uniform')

	if l_vocab_w == None:
	    self.n_f = n_f
	elif only_lexicalized:
	    self.n_f = n_f
	else:
	    self.n_f = n_f*2

	if model == 'ff':
	    self.convLayer = ConvLayer(self.embLayer.output,
	    	n_in=self.n_f,
	    	winDim = 4,
	    	n_out=n_hidden,
	    	activation = activation,
            	init = 'init_uniform')
            
	    if viterbi:
                self.outputLayer = Structured_OutputLayer( input=self.convLayer.output,
                                                    n_in=n_hidden,
                                                    n_out=l_vocab_out,
                                                    init = 'init_zeros')

	    else:
		self.outputLayer = Rec_OutputLayer( input=self.convLayer.output,
                                                    n_in=n_hidden,
                                                    n_out=l_vocab_out,
                                                    init = 'init_zeros')
		
            self.params = self.embLayer.params + self.convLayer.params + self.outputLayer.params
            
	else:
	    self.hiddenLayer = GRULayer(self.embLayer.output,
	    	n_in=self.n_f,
	    	n_out=n_hidden,
	    	#activation = activation,
            	init = 'init_uniform')

	    self.hiddenLayer_reverse = GRULayer(self.embLayer.output[:,::-1],
	    	n_in=self.n_f,
	    	n_out=n_hidden,
	    	#activation = activation,
            	init = 'init_uniform')
	
	    if viterbi:
		self.outputLayer = Structured_OutputLayer( input=T.concatenate([self.hiddenLayer.output,self.hiddenLayer_reverse.output], axis = 1),
            		n_in=n_hidden * 2,
            		n_out=l_vocab_out,
	    		init = 'init_uniform')
	    
	    else:
		self.outputLayer = Rec_OutputLayer( input=T.concatenate([self.hiddenLayer.output,self.hiddenLayer_reverse.output], axis = 1),
 	            	n_in=n_hidden * 2,
            		n_out=l_vocab_out,
	    		init = 'init_zeros')

	    self.params = self.embLayer.params + self.hiddenLayer.params + self.hiddenLayer_reverse.params + self.outputLayer.params

	self.trainer = globals()[trainer](self.params, lr)
        self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
        self.updates = self.trainer.get_updates(self.params, self.negative_log_likelihood(tags))
	     

	#Functions:
        self.train_perplexity = theano.function(inputs=[word_emb, word_len, word_beg, word_id, tags],
                                                    outputs=self.negative_log_likelihood(tags),
                                                    updates=self.updates,
                                                    allow_input_downcast=True,
                                                    on_unused_input='ignore')
            
        self.eval_perplexity = theano.function(inputs=[word_emb, word_len, word_beg, word_id, tags],
                		                           outputs=self.negative_log_likelihood(tags),
                	                                   allow_input_downcast=True,
                	                                   on_unused_input='ignore')
         
        self.predict = theano.function(inputs = [word_emb, word_len, word_beg, word_id],
                                           outputs=T.argmax(self.outputLayer.p_y_given_x, axis=1),
                                           allow_input_downcast=True,
                                           on_unused_input='ignore')

	self.output_params = theano.function(inputs = [],
        				     outputs = [ p for (p, wrt) in self.params ],
					     allow_input_downcast=True,
                                             on_unused_input='ignore')
	if viterbi:    
            self.output_decode =  theano.function(inputs = [word_emb, word_len, word_beg, word_id],
    			                           outputs= self.outputLayer.decode_forward(),
                	                           allow_input_downcast=True,
                	                           on_unused_input='ignore')

