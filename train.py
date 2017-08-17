#!/usr/bin/python
import numpy as np
import scipy
from read_data import READ
from pte_theano import PTE
from datetime import datetime
import logging

class train_pte(object):

	#Initialize graph
	def __init__(self):
		self.window_size=10
		self.graphs = READ(self.window_size) #Give window size as parameter
		self.graphs.generate_graphs()
		print("Read and graphing complete")
		self.ndims = 40
		self.lr = 0.05
		self.batch_size = 100
		self.k = 6
		self.nepochs = 1

	def train(self):
		p, nP, v1, v2, pd, v3, v4, pl, v5, v6  = self.graphs.gen_edgeprob()
		pte = PTE(self.graphs.nvertex, self.ndims, self.graphs.ndocs, self.graphs.nlabel)
		pte.ww_model()
		pte.wd_model()
		pte.wl_model()
		currentHour = datetime.utcnow().hour

		# setting up logger
		logger = logging.getLogger("wordTovec")
		logger.setLevel(logging.INFO)
		logger.setLevel(logging.DEBUG)
		fh = logging.FileHandler("word2graph2vec.log")
		formatter = logging.Formatter('%(asctime)s %(message)s')
		fh.setFormatter(formatter)
		logger.addHandler(fh)
		logger.info("Training started")
		logger.info("Total edges : %f " % self.graphs.nedge)

		for it in xrange(0, self.graphs.nedge, self.batch_size):
			sample = np.random.choice(p.shape[0], self.batch_size, p=p)
			k=0
			while k < sample.shape[0]:
				i = v1[sample[k]]-1
				j = v2[sample[k]]-1
				i_set = np.asarray(np.random.choice(self.graphs.nvertex, size=self.k, p=nP), dtype=np.int32)
				if i in i_set:
					i_set = np.delete(i_set, np.where(i_set==i))
				costww = pte.pretraining_ww(j, i, i_set)
				k = k + 1
			
			sample = np.random.choice(pd.shape[0], self.batch_size, p=pd)
			k=0
			while k < sample.shape[0]:
				i = v4[sample[k]]-1
				j = v3[sample[k]]-1
				i_set = np.asarray(np.random.choice(self.graphs.nvertex, size=self.k, p=nP), dtype=np.int32)
				if i in i_set:
					i_set = np.delete(i_set, np.where(i_set==i))
				costwd = pte.pretraining_wd(j, i, i_set)
				k = k+1

			sample = np.random.choice(pl.shape[0], self.batch_size, p=pl)
			k=0
			while k < sample.shape[0]:
				i = v6[sample[k]]-1	#one word
				j = v5[sample[k]]-1	#one label
				i_set = np.asarray(np.random.choice(self.graphs.nvertex, size=self.k, p=nP), dtype=np.int32)
				if i in i_set:
					i_set = np.delete(i_set, np.where(i_set==i))
				costwl = pte.pretraining_wl(j, i, i_set)
				k = k+1

			#print("Current it: ", it, " complete of total: ", self.graphs.nedge)
			if datetime.utcnow().hour >= currentHour+2:
				logger.info("ww Cost after 2 hrs training is %f" % costww)
				logger.info("wd Cost after 2 hrs training is %f" % costwd)
				logger.info("wl Cost after 2 hrs training is %f" % costwl)
				logger.info("Current it: %f " % it)
				logger.info("Saving the model")
				pte.save_model()
				currentHour += 2
		logger.info("Saving the model finally")
		pte.save_model()


if __name__ == "__main__":
	pte = train_pte()
	pte.train()


