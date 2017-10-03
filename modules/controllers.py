# -*- coding: utf-8 -*-
"""

Controllers for diffferent models

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
import models
import optimizers

#from scipy.optimize import minimize

dtype = theano.config.floatX


class ControlHawkesCTSM(object):
    # This is a seq 2 seq model train_er
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_end : T * size_batch -- T - t_i
        seq_time_to_current : T * T * size_batch --
        for each batch, it is T * T, and at each time step t,
        it tracks the ( t_i - t_i' ) for all t_i' < t_i
        seq_type_event : T * size_batch -- for each data
        and each time step, tracks the type of event k_i
        time_since_start_to_end : size_batch -- time for seq
        #
        seq_mask : T * size_batch -- 1/0
        seq_mask_to_current : T * T * size_batch -- 1/0
        '''
        self.seq_time_to_end = tensor.matrix(
            dtype=dtype, name='seq_time_to_end'
        )
        self.seq_time_to_current = tensor.tensor3(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_mask_to_current = tensor.tensor3(
            dtype=dtype, name='seq_mask_to_current'
        )
        #
        self.seq_sims_time_to_current = tensor.tensor3(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        self.seq_sims_mask_to_current = tensor.tensor3(
            dtype=dtype, name='seq_sims_mask_to_current'
        )
        #
        #
        self.hawkes_ctsm = models.HawkesCTSM(settings)
        #
        self.hawkes_ctsm.compute_loss(
            self.seq_time_to_end,
            self.seq_time_to_current,
            self.seq_type_event,
            self.time_since_start_to_end,
            self.seq_mask,
            self.seq_mask_to_current
        )
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(
                adam_params=None
            )
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(
                adam_params=None
            )
        else:
            print "Choose a optimizer ! "
        #
        if 'learn_rate' in settings:
            print "learn rate is set to : ", settings['learn_rate']
            self.adam_optimizer.set_learn_rate(
                settings['learn_rate']
            )
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = range(3)
        )
        #
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.time_since_start_to_end,
                self.seq_mask,
                self.seq_mask_to_current
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ],
            updates = self.adam_optimizer.updates
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.time_since_start_to_end,
                self.seq_mask,
                self.seq_mask_to_current
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ]
        )
        if settings['predict_lambda']:
            print "compiling dev function for intensity computation ... "
            self.hawkes_ctsm.compute_lambda(
                self.seq_type_event,
                self.seq_sims_time_to_current,
                self.seq_sims_mask,
                self.seq_sims_mask_to_current
            )
            self.model_dev_lambda = theano.function(
                inputs = [
                    self.seq_type_event,
                    self.seq_sims_time_to_current,
                    self.seq_sims_mask,
                    self.seq_sims_mask_to_current
                ],
                outputs = [
                    self.hawkes_ctsm.lambda_samples,
                    self.hawkes_ctsm.num_of_samples
                ]
            )
        #
        #self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #


class ControlHawkesInhibCTSM(object):
    # This is a seq 2 seq model train_er
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_end : T * size_batch -- T - t_i
        seq_time_to_current : T * T * size_batch --
        for each batch, it is T * T, and at each time step t,
        it tracks the ( t_i - t_i' ) for all t_i' < t_i
        seq_type_event : T * size_batch -- for each data
        and each time step, tracks the type of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- # of samples for seq
        #
        seq_mask : T * size_batch -- 1/0
        seq_mask_to_current : T * T * size_batch -- 1/0
        seq_sims_time_to_current : N * T * size_batch
        seq_sims_mask_to_current : N * T * size_batch
        seq_sims_mask : N * size_batch
        '''
        #self.seq_time_to_end = tensor.matrix(
        #    dtype=dtype, name='seq_time_to_end'
        #)
        self.seq_time_to_current = tensor.tensor3(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.num_sims_start_to_end = tensor.vector(
            dtype=dtype, name='num_sims_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_mask_to_current = tensor.tensor3(
            dtype=dtype, name='seq_mask_to_current'
        )
        self.seq_sims_time_to_current = tensor.tensor3(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_mask_to_current = tensor.tensor3(
            dtype=dtype, name='seq_sims_mask_to_current'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        #
        if settings['model'] == 'hawkesinhib':
            self.hawkes_ctsm = models.HawkesInhibCTSM(
                settings
            )
            list_constrain = [2]
        elif settings['model'] == 'hawkesinhibscale':
            self.hawkes_ctsm = models.HawkesInhibCTSM_scale(
                settings
            )
            list_constrain = [0, 3]
        else:
            print "called wrong controller"
        #
        #
        self.hawkes_ctsm.compute_loss(
            #self.seq_time_to_end,
            self.seq_time_to_current,
            self.seq_type_event,
            self.time_since_start_to_end,
            self.num_sims_start_to_end,
            self.seq_mask,
            self.seq_mask_to_current,
            self.seq_sims_time_to_current,
            self.seq_sims_mask_to_current,
            self.seq_sims_mask
        )
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(adam_params=None)
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(adam_params=None)
        else:
            print "Choose a optimizer ! "
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = list_constrain
        )
        #
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_mask_to_current,
                self.seq_sims_time_to_current,
                self.seq_sims_mask_to_current,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ],
            updates = self.adam_optimizer.updates
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_mask_to_current,
                self.seq_sims_time_to_current,
                self.seq_sims_mask_to_current,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ]
        )
        #
        #self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #


class ControlNeuralHawkesAdaptiveBaseCTSM_time(object):
    #
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        #self.seq_time_to_end = tensor.matrix(
        #    dtype=dtype, name='seq_time_to_end'
        #)
        self.seq_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        #self.seq_time_rep = tensor.tensor3(
        #    dtype=dtype, name='seq_time_rep'
        #)
        self.seq_time_values = tensor.matrix(
            dtype=dtype, name='seq_time_values'
        )
        #
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.num_sims_start_to_end = tensor.vector(
            dtype=dtype, name='num_sims_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_sims_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_index_in_hidden = tensor.imatrix(
            name='seq_sims_index_in_hidden'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        self.time_diffs = tensor.vector(
            dtype=dtype, name='time_diffs'
        )
        #
        #
        if settings['model'] == 'neuraladapttime':
            self.hawkes_ctsm = models.NeuralHawkesAdaptiveBaseCTSM_time(
                settings
            )
            list_constrain = []
        elif settings['model'] == 'neuraladapttimescale':
            self.hawkes_ctsm = models.NeuralHawkesAdaptiveBaseCTSM_time_scale(
                settings
            )
            list_constrain = [0]
        elif settings['model'] == 'neuralreduce':
            self.hawkes_ctsm = models.NeuralHawkesAdaptiveBaseCTSM_time_scale_r(
                settings
            )
            list_constrain = [0]
        elif settings['model'] == 'conttime':
            self.hawkes_ctsm = models.NeuralHawkesCTLSTM(
                settings
            )
            list_constrain = [0]
        else:
            print "called wrong controller"
        #
        assert(
            settings['loss_type'] == 'loglikehood' or settings['loss_type'] == 'prediction'
        )
        #
        if settings['loss_type'] == 'loglikehood':
            print "train with log-likelihood ... "
            self.hawkes_ctsm.compute_loss(
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                #self.seq_time_rep,
                self.seq_time_values,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            )
        else:
            print "train with prediction ... "
            #TODO: need to add switch for less memory
            #or faster speed
            #self.hawkes_ctsm.compute_prediction_loss(
            self.hawkes_ctsm.compute_prediction_loss_lessmem(
                self.seq_type_event,
                self.seq_time_values,
                self.seq_mask,
                self.time_diffs
            )
        #
        #self.hawkes_ctsm.compute_prediction(
        #    self.seq_type_event,
        #    self.seq_time_values,
        #    self.seq_mask,
        #    self.time_diffs
        #)
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(
                adam_params=None
            )
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(
                adam_params=None
            )
        else:
            print "Choose a optimizer ! "
        #
        if 'learn_rate' in settings:
            print "learn rate is set to : ", settings['learn_rate']
            self.adam_optimizer.set_learn_rate(
                settings['learn_rate']
            )
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = list_constrain
        )
        # in this version, no hard constraints on parameters
        #
        if settings['loss_type'] == 'loglikehood':
            print "optimize loglikehood ... "
            print "compiling training function ... "
            self.model_learn = theano.function(
                inputs = [
                    self.seq_time_to_current,
                    self.seq_type_event,
                    self.seq_time_values,
                    self.time_since_start_to_end,
                    self.num_sims_start_to_end,
                    self.seq_mask,
                    self.seq_sims_time_to_current,
                    self.seq_sims_index_in_hidden,
                    self.seq_sims_mask
                ],
                outputs = [
                    self.hawkes_ctsm.log_likelihood_seq,
                    self.hawkes_ctsm.log_likelihood_time,
                    self.hawkes_ctsm.log_likelihood_type,
                    self.hawkes_ctsm.num_of_events
                ],
                updates = self.adam_optimizer.updates,
                on_unused_input='ignore'
            )
            print "compiling dev function ... "
            self.model_dev = theano.function(
                inputs = [
                    #self.seq_time_to_end,
                    self.seq_time_to_current,
                    self.seq_type_event,
                    #self.seq_time_rep,
                    self.seq_time_values,
                    self.time_since_start_to_end,
                    self.num_sims_start_to_end,
                    self.seq_mask,
                    self.seq_sims_time_to_current,
                    self.seq_sims_index_in_hidden,
                    self.seq_sims_mask
                ],
                outputs = [
                    self.hawkes_ctsm.log_likelihood_seq,
                    self.hawkes_ctsm.log_likelihood_time,
                    self.hawkes_ctsm.log_likelihood_type,
                    self.hawkes_ctsm.num_of_events
                ],
                on_unused_input='ignore'
            )
            if settings['predict_lambda']:
                print "compiling dev function for intensity computation ... "
                self.hawkes_ctsm.compute_lambda(
                    self.seq_type_event,
                    self.seq_time_values,
                    self.seq_sims_time_to_current,
                    self.seq_sims_index_in_hidden,
                    self.seq_sims_mask
                )
                self.model_dev_lambda = theano.function(
                    inputs = [
                        self.seq_type_event,
                        self.seq_time_values,
                        self.seq_sims_time_to_current,
                        self.seq_sims_index_in_hidden,
                        self.seq_sims_mask
                    ],
                    outputs = [
                        self.hawkes_ctsm.lambda_samples,
                        self.hawkes_ctsm.num_of_samples
                    ],
                    on_unused_input='ignore'
                )
        else:
            print "optimize prediction ... "
            print "compiling training function ... "
            self.model_learn = theano.function(
                inputs = [
                    self.seq_type_event,
                    self.seq_time_values,
                    self.seq_mask,
                    self.time_diffs
                ],
                outputs = [
                    self.hawkes_ctsm.log_likelihood_type_predict,
                    self.hawkes_ctsm.num_of_errors,
                    self.hawkes_ctsm.square_errors,
                    self.hawkes_ctsm.num_of_events
                    #self.hawkes_ctsm.abs_grad_params
                ],
                updates = self.adam_optimizer.updates,
                on_unused_input='ignore'
            )
            print "compiling dev function ... "
            self.model_dev = theano.function(
                inputs = [
                    self.seq_type_event,
                    self.seq_time_values,
                    self.seq_mask,
                    self.time_diffs
                ],
                outputs = [
                    self.hawkes_ctsm.log_likelihood_type_predict,
                    self.hawkes_ctsm.num_of_errors,
                    self.hawkes_ctsm.square_errors,
                    self.hawkes_ctsm.num_of_events
                    #self.hawkes_ctsm.abs_grad_params
                    #
                ],
                on_unused_input='ignore'
            )
        #
        #
        self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# deprecated models
# TODO: modules below are deprecated
# they are models that we tried over this project
# most of them work, better than Hawkes baseline
# but still lose to our neural Hawkes with continuous-time LSTM


class ControlNeuralHawkesCTSM(object):
    #
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        #self.seq_time_to_end = tensor.matrix(
        #    dtype=dtype, name='seq_time_to_end'
        #)
        self.seq_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        self.seq_time_rep = tensor.tensor3(
            dtype=dtype, name='seq_time_rep'
        )
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.num_sims_start_to_end = tensor.vector(
            dtype=dtype, name='num_sims_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_sims_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_index_in_hidden = tensor.imatrix(
            name='seq_sims_index_in_hidden'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        #
        self.hawkes_ctsm = models.NeuralHawkesCTSM(settings)
        #
        self.hawkes_ctsm.compute_loss(
            #self.seq_time_to_end,
            self.seq_time_to_current,
            self.seq_type_event,
            self.seq_time_rep,
            self.time_since_start_to_end,
            self.num_sims_start_to_end,
            self.seq_mask,
            self.seq_sims_time_to_current,
            self.seq_sims_index_in_hidden,
            self.seq_sims_mask
        )
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(adam_params=None)
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(adam_params=None)
        else:
            print "Choose a optimizer ! "
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = [1]
        )
        #
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.seq_time_rep,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ],
            updates = self.adam_optimizer.updates
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.seq_time_rep,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ]
        )
        #
        #self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #
    #


class ControlGeneralizedNeuralHawkesCTSM(object):
    #
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        #self.seq_time_to_end = tensor.matrix(
        #    dtype=dtype, name='seq_time_to_end'
        #)
        self.seq_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        self.seq_time_rep = tensor.tensor3(
            dtype=dtype, name='seq_time_rep'
        )
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.num_sims_start_to_end = tensor.vector(
            dtype=dtype, name='num_sims_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_sims_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_index_in_hidden = tensor.imatrix(
            name='seq_sims_index_in_hidden'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        #
        self.hawkes_ctsm = models.GeneralizedNeuralHawkesCTSM(
            settings
        )
        #
        self.hawkes_ctsm.compute_loss(
            #self.seq_time_to_end,
            self.seq_time_to_current,
            self.seq_type_event,
            self.seq_time_rep,
            self.time_since_start_to_end,
            self.num_sims_start_to_end,
            self.seq_mask,
            self.seq_sims_time_to_current,
            self.seq_sims_index_in_hidden,
            self.seq_sims_mask
        )
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(
                adam_params=None
            )
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(
                adam_params=None
            )
        else:
            print "Choose a optimizer ! "
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = []
        )
        # in this version, no hard constraints on parameters
        #
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.seq_time_rep,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ],
            updates = self.adam_optimizer.updates
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.seq_time_rep,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ]
        )
        #
        #self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #
    #


class ControlNeuralHawkesAdaptiveBaseCTSM(object):
    #
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        #self.seq_time_to_end = tensor.matrix(
        #    dtype=dtype, name='seq_time_to_end'
        #)
        self.seq_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        self.seq_time_rep = tensor.tensor3(
            dtype=dtype, name='seq_time_rep'
        )
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.num_sims_start_to_end = tensor.vector(
            dtype=dtype, name='num_sims_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_sims_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_index_in_hidden = tensor.imatrix(
            name='seq_sims_index_in_hidden'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        #
        self.hawkes_ctsm = models.NeuralHawkesAdaptiveBaseCTSM(
            settings
        )
        #
        self.hawkes_ctsm.compute_loss(
            #self.seq_time_to_end,
            self.seq_time_to_current,
            self.seq_type_event,
            self.seq_time_rep,
            self.time_since_start_to_end,
            self.num_sims_start_to_end,
            self.seq_mask,
            self.seq_sims_time_to_current,
            self.seq_sims_index_in_hidden,
            self.seq_sims_mask
        )
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(
                adam_params=None
            )
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(
                adam_params=None
            )
        else:
            print "Choose a optimizer ! "
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = []
        )
        # in this version, no hard constraints on parameters
        #
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.seq_time_rep,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ],
            updates = self.adam_optimizer.updates
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                self.seq_time_rep,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ]
        )
        #
        #self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #
    #



class ControlNeuralHawkesCTSM_time(object):
    #
    '''
    this model stems from neural hawkes
    but encode time (positive real values) with neural nodes
    '''
    #
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        #self.seq_time_to_end = tensor.matrix(
        #    dtype=dtype, name='seq_time_to_end'
        #)
        self.seq_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        #self.seq_time_rep = tensor.tensor3(
        #    dtype=dtype, name='seq_time_rep'
        #)
        self.seq_time_values = tensor.matrix(
            dtype=dtype, name='seq_time_values'
        )
        #
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.num_sims_start_to_end = tensor.vector(
            dtype=dtype, name='num_sims_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_sims_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_index_in_hidden = tensor.imatrix(
            name='seq_sims_index_in_hidden'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        #
        self.hawkes_ctsm = models.NeuralHawkesCTSM_time(settings)
        #
        self.hawkes_ctsm.compute_loss(
            #self.seq_time_to_end,
            self.seq_time_to_current,
            self.seq_type_event,
            #self.seq_time_rep,
            self.seq_time_values,
            self.time_since_start_to_end,
            self.num_sims_start_to_end,
            self.seq_mask,
            self.seq_sims_time_to_current,
            self.seq_sims_index_in_hidden,
            self.seq_sims_mask
        )
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(adam_params=None)
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(adam_params=None)
        else:
            print "Choose a optimizer ! "
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = [1]
        )
        #
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                #self.seq_time_rep,
                self.seq_time_values,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ],
            updates = self.adam_optimizer.updates
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                #self.seq_time_rep,
                self.seq_time_values,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ]
        )
        #
        #self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #
    #


class ControlGeneralizedNeuralHawkesCTSM_time(object):
    #
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        #self.seq_time_to_end = tensor.matrix(
        #    dtype=dtype, name='seq_time_to_end'
        #)
        self.seq_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        #self.seq_time_rep = tensor.tensor3(
        #    dtype=dtype, name='seq_time_rep'
        #)
        self.seq_time_values = tensor.matrix(
            dtype=dtype, name='seq_time_values'
        )
        #
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.num_sims_start_to_end = tensor.vector(
            dtype=dtype, name='num_sims_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_sims_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_index_in_hidden = tensor.imatrix(
            name='seq_sims_index_in_hidden'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        #
        self.hawkes_ctsm = models.GeneralizedNeuralHawkesCTSM_time(
            settings
        )
        #
        self.hawkes_ctsm.compute_loss(
            #self.seq_time_to_end,
            self.seq_time_to_current,
            self.seq_type_event,
            #self.seq_time_rep,
            self.seq_time_values,
            self.time_since_start_to_end,
            self.num_sims_start_to_end,
            self.seq_mask,
            self.seq_sims_time_to_current,
            self.seq_sims_index_in_hidden,
            self.seq_sims_mask
        )
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(
                adam_params=None
            )
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(
                adam_params=None
            )
        else:
            print "Choose a optimizer ! "
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = []
        )
        # in this version, no hard constraints on parameters
        #
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                #self.seq_time_rep,
                self.seq_time_values,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ],
            updates = self.adam_optimizer.updates
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                #self.seq_time_to_end,
                self.seq_time_to_current,
                self.seq_type_event,
                #self.seq_time_rep,
                self.seq_time_values,
                self.time_since_start_to_end,
                self.num_sims_start_to_end,
                self.seq_mask,
                self.seq_sims_time_to_current,
                self.seq_sims_index_in_hidden,
                self.seq_sims_mask
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_seq,
                self.hawkes_ctsm.log_likelihood_time,
                self.hawkes_ctsm.log_likelihood_type,
                self.hawkes_ctsm.num_of_events
            ]
        )
        #
        #self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #
    #
#
#
#
#
#
