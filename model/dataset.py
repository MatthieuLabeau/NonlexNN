# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import linecache
import codecs
import random

# --- Processing Data : corpus ---
"""
Class taking as inputs paths to text and corresponding tag files, with the vocabularies, to create a 'sampler' that will output 
the necessary information for the network, sentence by sentence:
n-grams of characters indexes
index of the tag
length of each word
position of the characters that start each word
word indexes
Sentences are shuffled
The last input indicates the level of detail for the tags : 'ups' is the simplest, 'pos' is Part-of-Speeach, 'mph' contains all the morpho syntaxic informations.
"""
class dataset():
    def __init__(self, path_to_data_file, path_to_output_file, wordvocab, charvocab, outvocab, batch_size = 1, winDim = 5, diff = "mph"):

	self.infile = path_to_data_file
	self.outputfile = path_to_output_file
	self.wordvocab = wordvocab
	self.charvocab = charvocab
	self.outvocab = outvocab

        self.batch_size = batch_size
	self.x_chars = list()
	self.y = list()
        self.wl = list()
        self.wb = list()
	self.wid = list()

        with open(self.infile) as data_file: 
 	    with open(self.outputfile) as tags_file:
		for line_d, line_t in zip(data_file, tags_file):

		    words = [ w for w in line_d.strip().split()	]
		    if len(words) > 1:
		        labels = line_t.strip().split()
		        assert(len(words)==len(labels))
		        nwords = len(words)
	    	        w_chars = list()
            	        w_l = list()
            	        w_b = list()
		        w_id = list()
		        for w in words:
			    w_id.append(self.wordvocab.get(w,0))
                	    w = winDim*["bow"]+list(w)+winDim*["eow"]  
                	    w_b.append(sum(w_l))
			    w_chars.extend([[self.charvocab.get(char,0) for char in w[i:i+winDim]] for i in range(len(w)-winDim+1)])
			    w_l.append(len(w)-winDim+1)
            	        self.wb.append(w_b)
            	        self.wl.append(w_l)
		        self.wid.append(w_id)
           	        self.x_chars.append(np.asarray(w_chars,dtype = 'int32'))
            	        if diff == "mph" or diff == "pos" or diff == "ups":
            		    pids = [self.outvocab.get(p,0) for p in labels]
	                    self.y.append(np.asarray([pids[i] for i in range(nwords)],dtype='int32'))	
	    	        elif diff == "factored":
			    pdec = [ p.split("-") for p in labels]
                            cont = list()
                            for k in range(nwords):
                                cont.append(list())
                            for k,p in enumerate(pdec):
			        for l,f in enumerate(p):
				    cont[k].append((self.outvocab[l])[f])
		    		
			    self.y.append(np.asarray([np.asarray([cont[i][j] for j in range(8)]) for i in range(nwords)],dtype='int32'))
		    
		        assert(len(self.x_chars)==len(self.y))
        
	self.cpt=0
	self.tot=len(self.x_chars)
	self.ids=range(self.tot)
        random.shuffle(self.ids)	

    def sampler(self):
        while True:
            if (self.cpt+self.batch_size > self.tot):
                self.cpt=0
                random.shuffle(self.ids)
            xlist = list()
            ylist = list()
            llist = list()
            blist = list()
	    idlist = list()
            for i in range(self.batch_size):
                self.cpt+=1
                xlist.append(self.x_chars[self.ids[self.cpt-1]])
                ylist.append(self.y[self.ids[self.cpt-1]])
                llist.append(self.wl[self.ids[self.cpt-1]])
                blist.append(self.wb[self.ids[self.cpt-1]])
		idlist.append(self.wid[self.ids[self.cpt-1]])
            yield xlist, ylist, llist, blist, idlist
