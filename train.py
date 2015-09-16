# -*- coding: utf-8 -*-

from model import *
from model.data_proc import *
from model.dataset import dataset
from model.model import *

# --- Params ---
N = 300001 #Number of sentences fed to the network for training. Training set is made of 40472 sentences, dev and test set, 5.000 sentences each.
M = 100000 #The script makes an evaluation of the model every M sentences
winDim = 5 # Size of the character n-grams
n_char = 100 #Dimension of character embeddings
n_f = 200 #Dimension of word embeddings
n_hidden = 200 #Dimension of hidden representations
model = 'ff' #Model used for classification : 'ff' or 'biRNN'
char_model = 'conv' #Model used for word encoding : 'conv' or 'RNN'
viterbi = False #Structured output using Viterbi ?
lexicalized = False #Use of lexicalized word embeddings to concatenate with embeddings from characters ? 
lexicalized_threshold = 1 #Threshold on count to appear in the word vocabulary
trainer = 'AdagradTrainer' #Which trainer to use ? 
lr = 0.05 #With what initial learning rate? 
diff = "pos" # Tag granularity : ups - 12 / pos - 54 / mph - 619 en train, 629 total 
batch_size = 1
only_lexicalized = False #Only use lexicalized word embeddings - 'lexicalized' needs to be True 


import logging
import datetime

now = datetime.datetime.now()
logging.basicConfig(filename='logs/conv_%s.log' % now.isoformat(), filemode='w', level=logging.DEBUG)

logging.info('winDim : %i' % winDim)
logging.info('n_char : %i' % n_char)
logging.info('n_f : %i' % n_f)
logging.info('n_hidden : %i' % n_hidden)
logging.info('classification_model: %s' % model)
logging.info('char_embeddings_model: %s' % char_model)
logging.info('Lexicalized (with frequency threshold): %s, %i' % (lexicalized, lexicalized_threshold))
logging.info('Tag recursion: %s' % viterbi)
logging.info('Only lexicalized ?: %s' % only_lexicalized)
logging.info('trainer: %s' % trainer)


# --- Paths for the data ---
path = "data/"

train_words = "train.wrd"
dev_words = "devel.wrd"
test_words = "test.wrd"

train_pos = "train." + diff
dev_pos = "devel." + diff
test_pos = "test." + diff

""" Creating the vocabularies """
wordvocab, freq_w = create_vocab(path + train_words + ".voc", threshold = lexicalized_threshold)
charvocab = create_charvocab(path + train_words)
outvocab, freq_t = create_vocab(path + train_pos + ".voc")

""" Creating the datasets and their samplers """
trainset = dataset(path + train_words, path + train_pos, wordvocab, charvocab, outvocab, batch_size, winDim, diff)
devset = dataset(path + dev_words, path + dev_pos, wordvocab, charvocab, outvocab, batch_size, winDim, diff)
testset = dataset(path + test_words, path + test_pos, wordvocab, charvocab, outvocab, batch_size, winDim, diff)

sampler = trainset.sampler()
dev_sampler = devset.sampler()
test_sampler = testset.sampler()

l_vocab = len(charvocab)
if lexicalized:
    l_vocab_w = len(wordvocab)
else:
    l_vocab_w = None
l_vocab_out = len(outvocab)

#Model init
""" Creating the network """
Conv = Conv_model(l_vocab_w=l_vocab_w, l_vocab=l_vocab, l_vocab_out=l_vocab_out, winDim=winDim, n_char=n_char, n_f=n_f, n_hidden=n_hidden, lr = lr, trainer = trainer, activation=T.tanh, model = model, viterbi = viterbi, char_model = char_model, only_lexicalized = only_lexicalized )

logging.info('Learning rate: %f' % Conv.trainer.lr.eval())
logging.info('e: %f' % Conv.trainer.e.eval())

#Training Loop
train_res = []

for i in range(N):

    #Training
    inputs, tags, llen, beg, words = sampler.next()

    res = Conv.train_perplexity(inputs[0], llen[0], beg[0], words[0], tags[0])
    train_res.append(res)

    #Evaluation on dev and test
    if (i % M) == 0:

	dev_res=[]
        dev_pred=[]
        dev_tags=[]
        dev_len=[]
	dev_words=[]
	
	test_res=[]
        test_pred=[]
        test_tags=[]
        test_len=[]
	test_words=[]

	if viterbi:
	    dev_pred_v=[]
            test_pred_v=[]

        for j in range(5000):
	    #dev
            inputs, tags, llen, beg, words = dev_sampler.next()

	    res = Conv.eval_perplexity(inputs[0], llen[0], beg[0], words[0], tags[0])
            pred = Conv.predict(inputs[0], llen[0], beg[0], words[0])

	    if viterbi:
                viterbi_max, viterbi_argmax =  Conv.output_decode(inputs[0], llen[0], beg[0], words[0])
                first_ind = np.argmax(viterbi_max[-1])
                viterbi_pred =  backtrack(first_ind, viterbi_argmax)
	        dev_pred_v.extend(viterbi_pred)
	
	    dev_pred.extend(pred)
            dev_tags.extend(tags[0])
            dev_res.append(res)
            dev_len.append(len(pred))
	    dev_words.extend(words[0])

            #test
	    inputs, tags, llen, beg, words = test_sampler.next()

	    res = Conv.eval_perplexity(inputs[0], llen[0], beg[0], words[0], tags[0])
            pred = Conv.predict(inputs[0], llen[0], beg[0], words[0])

	    if viterbi:
                viterbi_max, viterbi_argmax =  Conv.output_decode(inputs[0], llen[0], beg[0], words[0])
                first_ind = np.argmax(viterbi_max[-1])
                viterbi_pred =  backtrack(first_ind, viterbi_argmax)
	        test_pred_v.extend(viterbi_pred)

            test_pred.extend(pred)
            test_tags.extend(tags[0])
            test_res.append(res)
            test_len.append(len(pred))
	    test_words.extend(words[0])	    

        logging.info("Test : After %i examples" % i)
	logging.info("Average training cross-entropy on examples seen since last validation: %f " % np.mean(train_res))
	train_res = []    
        diff_dev = [(i,j, w) for i,j, w in zip(dev_pred ,dev_tags, dev_words) if i != j]
	diff_test = [(i,j, w) for i,j, w in zip(test_pred ,test_tags, test_words) if i != j]
	value_dev = float(len(diff_dev))/sum(dev_len)
        value_test = float(len(diff_test))/sum(test_len)
	if viterbi:
            diff_dev_v = [(i,j, w) for i,j, w in zip(dev_pred_v,dev_tags, dev_words) if i != j]
            diff_test_v = [(i,j, w) for i,j, w in zip(test_pred_v,test_tags, test_words) if i != j]
            value_dev_v = float(len(diff_dev_v))/sum(dev_len)
            value_test_v = float(len(diff_test_v))/sum(test_len)
	logging.info("Dev set cross-entropy : %f" % np.mean(dev_res))
        logging.info("Error rate for tag prediction on dev set : %f" % value_dev )
	if viterbi:
	    logging.info("Error rate for tag prediction on dev set - using viterbi decoding : %f" % value_dev_v )
	logging.info("Test set cross-entropy : %f" % np.mean(test_res))
        logging.info("Error rate for tag prediction on test set : %f" % value_test )
	if viterbi:
	    logging.info("Error rate for tag prediction on test set - using viterbi decoding: %f" % value_test_v )

