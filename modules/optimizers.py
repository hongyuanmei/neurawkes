# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8th 10:58:37 2016

useful optimizers

@author: hongyuan
"""

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
#import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import utils

dtype=theano.config.floatX

class SGD(object):
    # Adam optimizer
    def __init__(self, adam_params=None):
        print "creating SGD optimizer ... "
        # set hyper params, and use the values in paper as default
        if adam_params == None:
            self.alpha = theano.shared(numpy.float32(1e-3), 'alpha')
            self.t_step = theano.shared(numpy.float32(1), 't_step')
        else:
            self.alpha = adam_params['alpha']
            self.t_step = adam_params['t_step']
        self.updates = []
        #self.m_params, self.v_params = [], []
    #
    def set_learn_rate(self, learn_rate):
        self.alpha = theano.shared(
            numpy.float32(learn_rate), 'alpha'
        )
    #
    def compute_updates(
        self, params, grad_params, list_constrain = []
    ):
        print "computing updates ... "
        for idx_param, (param, grad_param) in enumerate(zip(params, grad_params) ):
            param_t = param - (
                (self.alpha / self.t_step) + numpy.float32(1e-6)
            ) * grad_param
            if idx_param in list_constrain:
                param_t_positive = param_t + abs(param_t)
                param_t_positive *= numpy.float32(0.5)
                param_t_positive += numpy.float32(1e-9)
                self.updates.append( (param, param_t_positive) )
            else:
                self.updates.append( (param, param_t) )
            #
        self.updates.append( (self.t_step, self.t_step+1.0) )
        print "updates computed ! "


class Adam(object):
    # Adam optimizer
    def __init__(self, adam_params=None):
        print "creating Adam optimizer ... "
        # set hyper params, and use the values in paper as default
        if adam_params == None:
            self.alpha = theano.shared(numpy.float32(1e-3), 'alpha')
            #self.alpha = theano.shared(numpy.float32(1), 'alpha')
            #
            self.beta_1 = theano.shared(numpy.float32(0.9), 'beta_1')
            self.beta_2 = theano.shared(numpy.float32(0.999), 'beta_2')
            self.eps = theano.shared(numpy.float32(1e-8), 'eps')
            self.decay = theano.shared(numpy.float32(1.0-1e-8), 'decay')
            #self.decay = theano.shared(numpy.float32(1.0-1e-16), 'decay')
            self.t_step = theano.shared(numpy.float32(1), 't_step')
            self.beta_t = theano.shared(numpy.float32(0.9), 'beta_t')
        else:
            self.alpha = adam_params['alpha']
            self.beta_1 = adam_params['beta_1']
            self.beta_2 = adam_params['beta_2']
            self.eps = adam_params['eps']
            self.decay = adam_params['decay']
            self.t_step = adam_params['t_step']
            self.beta_t = adam_params['beta_t']
        self.updates = []
        self.m_params, self.v_params = [], []
        self.clip = False
        self.clip_min = numpy.float32(-5.0)
        self.clip_max = numpy.float32(5.0)
    #
    def set_learn_rate(self, learn_rate):
        print "setting learning rate : ", learn_rate
        self.alpha = theano.shared(
            numpy.float32(learn_rate), 'alpha'
        )
    #
    def compute_updates(
        self, params, grad_params, list_constrain = []
    ):
        print "computing updates ... "
        for param in params:
            param_shape = numpy.shape(param.get_value())
            self.m_params.append(
                theano.shared(numpy.zeros(param_shape, dtype=dtype))
            )
            self.v_params.append(
                theano.shared(numpy.zeros(param_shape, dtype=dtype))
            )
        for idx_param, (param, grad_param, m_param, v_param) in enumerate(zip(params, grad_params, self.m_params, self.v_params) ):
            #
            if self.clip:
                grad_param = tensor.clip(
                    grad_param, self.clip_min, self.clip_max
                )
            #
            m_0 = self.beta_t * m_param + (1-self.beta_t)*grad_param
            v_0 = self.beta_2 * v_param + (1-self.beta_2)*(grad_param**2)
            m_t = m_0 / (1-(self.beta_1**self.t_step))
            v_t = v_0 / (1-(self.beta_2**self.t_step))
            param_t = param - self.alpha*( m_t / (tensor.sqrt(v_t)+self.eps) )
            #
            if idx_param in list_constrain:
                param_t_positive = param_t + abs(param_t)
                param_t_positive *= numpy.float32(0.5)
                param_t_positive += numpy.float32(1e-9)
                self.updates.append(
                    (param, param_t_positive)
                )
            else:
                self.updates.append( (param, param_t) )
            #
            self.updates.append( (m_param, m_0) )
            self.updates.append( (v_param, v_0) )
        self.updates.append( (self.t_step, self.t_step+1.0) )
        self.updates.append( (self.beta_t, self.beta_t*self.decay) )
        print "updates computed ! "
