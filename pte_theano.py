#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from theano import tensor as T
import theano
import numpy as np

class PTE(object):
    '''
    Defines the PTE model (cost function, parameters in theano.
    '''
    def __init__(self, nvertex, out_dim, ndocs, nlabels, lr=0.05):
        '''
        Parameters specs:
            nvertex : no of vertices in the graph
            out_dim : node vector dimension
            ndocs : number of documents in the corpus
            nlabels : number of labels
            lr : learning rate.
        '''
        # TO-DO: Try initialization from uniform
        # Initialize model paramters
        eps = np.sqrt(1.0 / float(out_dim))
        self.w = np.asarray(np.random.uniform(low=-eps, high=eps, size=(nvertex, out_dim)),
                       dtype=theano.config.floatX)
        self.d = np.asarray(np.random.uniform(low=-eps, high=eps, size=(ndocs, out_dim)),
                       dtype=theano.config.floatX)
        self.l = np.asarray(np.random.uniform(low=-eps, high=eps, size=(nlabels, out_dim)),
                       dtype=theano.config.floatX)
        self.W = theano.shared(self.w, name='W', borrow=True)
        self.D = theano.shared(self.d, name='D', borrow=True)
        self.L = theano.shared(self.l, name='L', borrow=True)
        self.lr = lr

    def ww_model(self):
        '''
        Performs SGD update (pre-training on ww graph).
        '''
        indm = T.iscalar()
        indc = T.iscalar()
        indr = T.ivector()		#vector of 5 negative edge samples
        Uj = self.W[indm, :] #one row of W
        Ui = self.W[indc, :] #one row of W
        Ui_Set = self.W[indr, :] #rows for negative edge sampling
        cost_ww = T.log(T.nnet.sigmoid(T.dot(Ui, Uj)))
        cost_ww -= T.log(T.sum(T.nnet.sigmoid(T.sum(Uj * Ui_Set, axis=1))))
        #cost_ww = T.dot(Ui, Uj)
        #cost_ww -= T.log(T.sum(T.exp(T.sum(Uj * Ui_Set, axis=1))))
        cost = -cost_ww
        grad_ww = T.grad(cost, [Uj, Ui, Ui_Set]) #gradient w.r.t 3 variables
        deltaW = T.inc_subtensor(self.W[indm, :], - (self.lr) * grad_ww[0])
        deltaW = T.inc_subtensor(deltaW[indc, :], - (self.lr) * grad_ww[1])
        deltaW = T.inc_subtensor(deltaW[indr, :], - (self.lr) * grad_ww[2])
        updateD = [(self.W, deltaW)]
        self.train_ww = theano.function(inputs=[indm, indc, indr], outputs=cost, updates=updateD)

    def pretraining_ww(self, indm, indc, indr):
        return self.train_ww(indm, indc, indr)

    def wd_model(self):
        '''
        Performs SGD update (pre-training on wd graph).
        '''
        indm = T.iscalar()
        indc = T.iscalar()
        indr = T.ivector()		#vector of 5 negative edge samples
        Uj = self.D[indm, :] #one row of D
        Ui = self.W[indc, :] #one row of W
        Ui_Set = self.W[indr, :] #rows of W for negative edge sampling
        cost_wd = T.log(T.nnet.sigmoid(T.dot(Ui, Uj)))
        cost_wd -= T.log(T.sum(T.nnet.sigmoid(T.sum(Uj * Ui_Set, axis=1))))
        cost = -cost_wd
        grad_wd = T.grad(cost, [Uj, Ui, Ui_Set]) #gradient w.r.t 3 variables
        
        deltaD = T.inc_subtensor(self.D[indm, :], - (self.lr) * grad_wd[0])
        deltaW = T.inc_subtensor(self.W[indc, :], - (self.lr) * grad_wd[1])
        deltaW = T.inc_subtensor(deltaW[indr, :], - (self.lr) * grad_wd[2])
        updateD = [(self.W, deltaW), (self.D, deltaD)]
        self.train_wd = theano.function(inputs=[indm, indc, indr], outputs=cost, updates=updateD)

    def pretraining_wd(self, indm, indc, indr):
        return self.train_wd(indm, indc, indr)

    def wl_model(self):
        '''
        Performs SGD update (pre-training on wd graph).
        '''
        indm = T.iscalar()
        indc = T.iscalar()
        indr = T.ivector()		#vector of 5 negative edge samples
        Uj = self.L[indm, :] #one row of L
        Ui = self.W[indc, :] #one row of W
        Ui_Set = self.W[indr, :] #rows of W for negative edge sampling
        cost_wl = T.log(T.nnet.sigmoid(T.dot(Ui, Uj)))
        cost_wl -= T.log(T.sum(T.nnet.sigmoid(T.sum(Uj * Ui_Set, axis=1))))
        cost = -cost_wl
        grad_wl = T.grad(cost, [Uj, Ui, Ui_Set]) #gradient w.r.t 3 variables
        
        deltaL = T.inc_subtensor(self.L[indm, :], - (self.lr) * grad_wl[0])
        deltaW = T.inc_subtensor(self.W[indc, :], - (self.lr) * grad_wl[1])
        deltaW = T.inc_subtensor(deltaW[indr, :], - (self.lr) * grad_wl[2])
        updateD = [(self.W, deltaW), (self.L, deltaL)]
        self.train_wl = theano.function(inputs=[indm, indc, indr], outputs=cost, updates=updateD)

    def pretraining_wl(self, indm, indc, indr):
        return self.train_wl(indm, indc, indr)

    def save_model(self):
        '''
        Save embedding matrices on disk
        '''
        W = self.W.get_value()
        np.save('lookupW', W)

