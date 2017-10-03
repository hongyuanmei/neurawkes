# -*- coding: utf-8 -*-
"""

Here are the models
continuous-time sequential model (CTSM)

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

#
class HawkesCTSM(object):
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        print "initializing Hawkes CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            # initialize variables
            self.mu = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='mu'
            )
            '''
            we need to notice that: in these matrices of K * K
            the (i, j) entry is the effect of j-th to i-th
            this order may be changed in the neural hawkes
            for the sake of implementation ease or the convention of Theano
            '''
            self.alpha = theano.shared(
                numpy.ones(
                    (self.dim_process, self.dim_process),
                    dtype=dtype
                ), name='alpha'
            )
            self.delta = theano.shared(
                numpy.ones(
                    (self.dim_process, self.dim_process),
                    dtype=dtype
                ), name='delta'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.mu = theano.shared(
                model_pre_train['mu'], name='mu'
            )
            self.alpha = theano.shared(
                model_pre_train['alpha'], name='alpha'
            )
            self.delta = theano.shared(
                model_pre_train['delta'], name='delta'
            )
        #
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            self.mu, self.alpha, self.delta
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        # to evaluate per-event intensity predict
        # this should be filterd by mask
        self.lambda_samples = None
        self.num_of_samples = None
        #
    #
    #
    def compute_loss(
        self,
        seq_time_to_end, seq_time_to_current, seq_type_event,
        time_since_start_to_end,
        seq_mask, seq_mask_to_current
    ):
        '''
        use this function to compute negative log likelihood
        seq_time_to_end : T * size_batch -- T-t_i
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
        print "computing loss function of Hawkes model ... "
        # first compute the 3rd term in loss
        alpha_over_seq = self.alpha[
            :, seq_type_event
        ] # dim_process * T * size_batch
        delta_over_seq = self.delta[
            :, seq_type_event
        ] # dim_process * T * size_batch
        #
        term_3 = tensor.sum(
            tensor.sum(
                (
                    (
                        numpy.float32(1.0) - tensor.exp(
                            -delta_over_seq * seq_time_to_end[
                                None, :, :
                            ]
                        )
                    ) * alpha_over_seq / delta_over_seq
                ),
                axis = 0
            ) * seq_mask,
            axis = 0
        ) # (size_batch, )
        # then we compute the 2nd term
        term_2 = tensor.sum(self.mu) * time_since_start_to_end
        # (size_batch, )
        # then we compute the 1st term, which is the trickest
        # we use seq_time_to_current : T * T * size_batch
        # seq_mask_to_current : T * T * size_batch
        lambda_over_seq = self.mu[:, None, None] + tensor.sum(
            (
                seq_mask_to_current[None,:,:,:]
                * (
                    alpha_over_seq[:,None,:,:] * tensor.exp(
                        -delta_over_seq[:,None,:,:]
                        * seq_time_to_current[None,:,:,:]
                    )
                )
            )
            , axis=2
        ) # dim_process * T * size_batch
        #
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis=0
        ) # T * size_batch
        #
        # now we choose the right lambda for each step
        # by using seq_type_event : T * size_batch
        new_shape_0 = lambda_over_seq.shape[1]*lambda_over_seq.shape[2]
        new_shape_1 = lambda_over_seq.shape[0]
        #
        back_shape_0 = lambda_over_seq.shape[1]
        back_shape_1 = lambda_over_seq.shape[2]
        #
        lambda_target_over_seq = lambda_over_seq.transpose(
            (1,2,0)
        ).reshape(
            (
                new_shape_0, new_shape_1
            )
        )[
            tensor.arange(new_shape_0),
            seq_type_event.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
    #
    #
    def compute_lambda(
        self,
        seq_type_event,
        seq_sims_time_to_current,
        seq_sims_mask,
        seq_sims_mask_to_current
    ):
        '''
        use this function to compute intensity
        seq_type_event : T * size_batch -- for each data
        and each time step, tracks the type of event k_i
        these are only used for computing intensity estimation
        N is the # of MonteCarlo samples
        seq_sims_time_to_current : N * T * size_batch -- for each batch, and at each time step t, track t_i-t_i' for t_i'<t_i
        seq_sims_mask : N * size_batch
        seq_sims_mask_to_current : N * T * size_batch
        '''
        print "computing intensity ... "
        # first compute the 3rd term in loss
        alpha_over_seq = self.alpha[
            :, seq_type_event
        ] # dim_process * T * size_batch
        delta_over_seq = self.delta[
            :, seq_type_event
        ] # dim_process * T * size_batch
        #
        '''
        in this block, we compute intensity
        at sampled time
        '''
        #
        lambda_samples = self.mu[:,None,None] + tensor.sum(
            (
                seq_sims_mask_to_current[None,:,:,:] * (
                    alpha_over_seq[:,None,:,:] * tensor.exp(
                        -delta_over_seq[:,None,:,:] * seq_sims_time_to_current[None,:,:,:]
                    )
                )
            ), axis=2
        )
        # K * N * size_batch
        self.lambda_samples = lambda_samples * seq_sims_mask[None,:,:]
        self.num_of_samples = tensor.sum(seq_sims_mask)
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
#
#
# Note : _scale means : we use scaling parameter in transfer function
#
class HawkesInhibCTSM_scale(object):
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        print "initializing Hawkes CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            # initialize variables
            self.scale = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='scale'
            )
            #
            self.mu = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='mu'
            )
            '''
            we need to notice that: in these matrices of K * K
            the (i, j) entry is the effect of j-th to i-th
            this order may be changed in the neural hawkes
            for the sake of implementation ease or the convention of Theano
            '''
            self.alpha = theano.shared(
                numpy.ones(
                    (self.dim_process, self.dim_process),
                    dtype=dtype
                ), name='alpha'
            )
            self.delta = theano.shared(
                numpy.ones(
                    (self.dim_process, self.dim_process),
                    dtype=dtype
                ), name='delta'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.scale = theano.shared(
                model_pre_train['scale'], name='scale'
            )
            self.mu = theano.shared(
                model_pre_train['mu'], name='mu'
            )
            self.alpha = theano.shared(
                model_pre_train['alpha'], name='alpha'
            )
            self.delta = theano.shared(
                model_pre_train['delta'], name='delta'
            )
        #
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            self.scale, # scale parameter
            self.mu, self.alpha, self.delta
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #
    #
    def soft_relu(self, x):
        # x is a symbolic tensor
        # tensor[(x == 0).nonzeros()]
        #v_max = numpy.float32(1e9)
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        #a = tensor.switch(z>=v_max, v_max, z)
        #y[(x>=100.0).nonzeros()] = x[(x>=100.0).nonzeros()]
        #np.finfo(np.float32).max
        return z
    #
    #
    def soft_relu_scale(self, x):
        # x is symbolic tensor
        # usually last dim is dim_process
        # but in this model, 0-th dim is dim_process
        # this is important !
        x /= self.scale[:,None,None]
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        z *= self.scale[:,None,None]
        return z
    #
    #
    def compute_loss(
        self,
        seq_time_to_current, seq_type_event,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask, seq_mask_to_current,
        seq_sims_time_to_current,
        seq_sims_mask_to_current,
        seq_sims_mask
    ):
        '''
        use this function to compute negative log likelihood
        seq_time_to_end : T * size_batch -- T-t_i
        seq_time_to_current : T * T * size_batch --
        for each batch, it is T * T, and at each time step t,
        it tracks the ( t_i - t_i' ) for all t_i' < t_i
        seq_type_event : T * size_batch -- for each data
        and each time step, tracks the type of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        #
        seq_mask : T * size_batch -- 1/0
        seq_mask_to_current : T * T * size_batch -- 1/0
        #
        seq_sims_mask : N * size_batch -- 1/0
        '''
        print "computing loss function of Hawkes model ... "
        # first compute the 3rd term in loss
        alpha_over_seq = self.alpha[
            :, seq_type_event
        ] # dim_process * T * size_batch
        delta_over_seq = self.delta[
            :, seq_type_event
        ] # dim_process * T * size_batch
        #
        lambda_over_seq_sims_tilde = self.mu[:,None,None] + tensor.sum(
            (
                seq_sims_mask_to_current[None,:,:,:] * (
                    alpha_over_seq[:,None,:,:] * tensor.exp(
                        -delta_over_seq[:,None,:,:] * seq_sims_time_to_current[None,:,:,:]
                    )
                )
            ), axis=2
        )
        # dim_process * N * size_batch
        #
        lambda_over_seq_sims = self.soft_relu_scale(
            lambda_over_seq_sims_tilde
        )
        #
        # dim_process * N * size_batch
        #
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=0
        )
        # N * size_batch
        # mask the lambda of simulations
        lambda_sum_over_seq_sims *= seq_sims_mask
        #
        #
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        # (size_batch, )
        term_2 = numpy.float32(0.0)
        #
        '''
        for this model, the computation of term_3 follows the same procedure of term_1, since we need to estimate lambda(s_j), i.e, we need large N * T * size_batch tensors for (1) time to current; (2) mask for (1).
        then we can just follow the steps of term_1 to finish the integral estimation.
        correspondingly, we need to modify the data processors, to generate the big tensors
        '''
        # then we compute the 1st term, which is the trickest
        # we use seq_time_to_current : T * T * size_batch
        # seq_mask_to_current : T * T * size_batch
        lambda_over_seq_tilde = self.mu[:, None, None] + tensor.sum(
            (
                seq_mask_to_current[None,:,:,:]
                * (
                    alpha_over_seq[:,None,:,:] * tensor.exp(
                        -delta_over_seq[:,None,:,:]
                        * seq_time_to_current[None,:,:,:]
                    )
                )
            )
            , axis=2
        )
        # dim_process * T * size_batch
        #
        lambda_over_seq = self.soft_relu_scale(
            lambda_over_seq_tilde
        )
        #
        # dim_process * T * size_batch
        #
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis=0
        ) # T * size_batch
        # now we choose the right lambda for each step
        # by using seq_type_event : T * size_batch
        new_shape_0 = lambda_over_seq.shape[1]*lambda_over_seq.shape[2]
        new_shape_1 = lambda_over_seq.shape[0]
        #
        back_shape_0 = lambda_over_seq.shape[1]
        back_shape_1 = lambda_over_seq.shape[2]
        #
        lambda_target_over_seq = lambda_over_seq.transpose(
            (1,2,0)
        ).reshape(
            (
                new_shape_0, new_shape_1
            )
        )[
            tensor.arange(new_shape_0),
            seq_type_event.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
#
class NeuralHawkesCTLSTM(object):
    '''
    This model uses:
    Adative base rate, interaction and decay
    Continuous-time LSTM
    Scale parameter s_k for softrelu curvature adjustment
    Reduced version -- delta param is D * D, not D * D * K
    '''
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Neural Hawkes with Continuous-time LSTM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            # this is useless in cont-time lstm
            self.dim_model = settings['dim_model']
            # initialize variables
            self.scale = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='scale'
            )
            #
            # the 0-th axis -- self.dim_model
            # is for dot product with hidden units
            # dot(h, W_delta) --> delta of size:
            # dim_model * dim_process
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            #
            self.W_recur = theano.shared(
                utils.sample_weights(
                    2*self.dim_model, 7*self.dim_model
                ), name='W_recur'
            )
            '''
            2 input :
            event rep, hidden state
            7 outputs :
            4 regular LSTM gates
            2 -- input_bar and forget_bar gate
            1 -- cell memory decay gate
            '''
            self.b_recur = theano.shared(
                numpy.zeros(
                    (7*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.scale = theano.shared(
                model_pre_train['scale'], name='scale'
            )
            #
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            #
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.c_0_target = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0_target'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            #self.mu, #self.delta,
            self.scale, # scale parameter
            self.W_alpha,
            self.Emb_event,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        #
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        # tensor[(x == 0).nonzeros()]
        #v_max = numpy.float32(1e9)
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        #a = tensor.switch(z>=v_max, v_max, z)
        #y[(x>=100.0).nonzeros()] = x[(x>=100.0).nonzeros()]
        #np.finfo(np.float32).max
        return z
    #
    #
    def soft_relu_scale(self, x):
        # x is symbolic tensor
        # last dim is dim_process
        # this is important !
        x /= self.scale
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        z *= self.scale
        return z
    #
    #
    def rnn_unit(
        self,
        emb_event_im1, time_interval_im1,
        hidden_t_im1, cell_t_im1, cell_im1_target
    ):
        '''
        This LSTM unit is working in continuous-time
        What a regular LSTM does :
        Take h_{i-1}, and update to h_i
        What a CT-LSTM does :
        Take h(t_{i-1}), which decays to t_{i-1}
        Use it and upate to h_i
        h_i is then used to compute Hawkes params
        #
        input:
        emb_event_imt = x_{i-1}
        time_interval_i = t_i - t_{i-1}
        h(t_{i-1}) right before THIS update
        c(t_{i-1}) right before THIS update
        c_{i-1}_target before THIS update
        output: ( # stands for not output it )
        h(t_i) right before NEXT update at t_i
        c(t_i) right before NEXT update at t_i
        c_i_target over ( t_{i-1}, t_i ]
        #h_i = h( t_{i-1} <-- t ) right after THIS update
        c_i = c( t_{i-1} <-- t ) right after THIS update
        decay_rate over ( t_{i-1}, t_i ]
        gate_output over ( t_{i-1}, t_i ]
        '''
        #TODO: update LSTM state at t_{i-1}
        pre_transform = tensor.concatenate(
            [emb_event_im1, hidden_t_im1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        # 4 regular LSTM gates
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[
                :, 3*self.dim_model:4*self.dim_model
            ]
        )
        # 2 -- input_bar and forget_bar gates
        gate_input_target = tensor.nnet.sigmoid(
            post_transform[
                :, 4*self.dim_model:5*self.dim_model
            ]
        )
        gate_forget_target = tensor.nnet.sigmoid(
            post_transform[
                :, 5*self.dim_model:6*self.dim_model
            ]
        )
        # cell memory decay
        decay_cell = self.soft_relu(
            post_transform[
                :, 6*self.dim_model:
            ]
        )
        # size : size_batch * dim_model
        #TODO: decay cell memory
        cell_i = gate_forget * cell_t_im1 + gate_input * gate_pre_c
        cell_i_target = gate_forget_target * cell_im1_target + gate_input_target * gate_pre_c
        #
        cell_t_i = cell_i_target + (
            cell_i - cell_i_target
        ) * tensor.exp(
            -decay_cell * time_interval_im1[:, None]
        )
        hidden_t_i = gate_output * tensor.tanh(
            cell_t_i
        )
        #TODO: get the hidden state right after this update, which is used to compute Hawkes params
        hidden_i = gate_output * tensor.tanh(
            cell_i
        )
        return hidden_t_i, cell_t_i, cell_i_target, cell_i, decay_cell, gate_output
        #return hidden_t_i, cell_t_i, cell_i_target, hidden_i, cell_i, decay_cell, gate_output
        #
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_values : (T+1) * size_batch -- t_i - t_i-1 starting as 0.0 at BOS event
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        Warning: There is overlap between seq_time_values and seq_time_to_current, so in this function, we happen not to use both. So we need to put on_unused_input='warn' in theano.function to avoid error message.
        '''
        print "computing loss function of Neural Hawkes model with continuous-time LSTM ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        '''
        No need to pass time values through thresholds
        Use time_values directly
        '''
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        initial_cell_target_mat = tensor.outer(
            self.expand, self.c_0_target
        )
        # size_batch * dim_model
        # seq_emb_event and seq_time_values start with
        # a special BOS event -- ( K, 0.0 )
        # to initialize the h, c and \bar{c}
        '''
        seq_cell_target, seq_cell : cell right AFTER THIS occurrence, including BOS
        seq_decay_cell, seq_gate_output : decay and gates AFTER THIS and BEFORE NEXT
        seq_hidden_t, seq_cell_t : hidden and cell right BEFORE NEXT occurrence
        '''
        [seq_hidden_t, seq_cell_t, seq_cell_target, seq_cell, seq_decay_cell, seq_gate_output], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(
                    input=seq_emb_event[:-1, :, :],
                    taps=[0]
                ),
                dict(
                    input=seq_time_to_current,
                    taps=[0]
                )
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1]),
                dict(initial=initial_cell_target_mat, taps=[-1]),
                None, None, None
            ],
            non_sequences = None
        )
        # size of outputs of this scan :
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t), c(t), and decay(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        #
        shape_hidden = seq_cell_target.shape
        # [ T , size_batch , dim_model ]
        shape_sims_index = seq_sims_index_in_hidden.shape
        # [ N, size_batch ]
        #
        seq_cell_target_sims = seq_cell_target.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_cell_sims = seq_cell.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_decay_cell_sims = seq_decay_cell.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_gate_output_sims = seq_gate_output.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        #
        seq_cell_with_time_sims = seq_cell_target_sims + (
            seq_cell_sims - seq_cell_target_sims
        ) * tensor.exp(
            -seq_decay_cell_sims * seq_sims_time_to_current[:, :, None]
        )
        seq_hidden_with_time_sims = seq_gate_output_sims * tensor.tanh(
            seq_cell_with_time_sims
        )
        #
        lambda_over_seq_sims_tilde = tensor.tensordot(
            seq_hidden_with_time_sims, self.W_alpha,
            (2, 0)
        )
        lambda_over_seq_sims = self.soft_relu_scale(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        #
        term_2 = numpy.float32(0.0)
        #
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_t : T * size_batch * dim_model
        #
        lambda_over_seq_tilde = tensor.tensordot(
            seq_hidden_t, self.W_alpha,
            (2, 0)
        )
        lambda_over_seq = self.soft_relu_scale(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        #
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def compute_lambda(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute intensity
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        initial_cell_target_mat = tensor.outer(
            self.expand, self.c_0_target
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden_t, seq_cell_t, seq_cell_target, seq_cell, seq_decay_cell, seq_gate_output], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(
                    input=seq_emb_event[:-1, :, :],
                    taps=[0]
                ),
                dict(
                    input=seq_time_values[1:, :],
                    taps=[0]
                )
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1]),
                dict(initial=initial_cell_target_mat, taps=[-1]),
                None, None, None
            ],
            non_sequences = None
        )
        #
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_cell_target.shape
        # [ T, size_batch, dim_model ]
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_cell_target_sims = seq_cell_target.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_cell_sims = seq_cell.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_decay_cell_sims = seq_decay_cell.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_gate_output_sims = seq_gate_output.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        #
        seq_cell_with_time_sims = seq_cell_target_sims + (
            seq_cell_sims - seq_cell_target_sims
        ) * tensor.exp(
            -seq_decay_cell_sims * seq_sims_time_to_current[:, :, None]
        )
        seq_hidden_with_time_sims = seq_gate_output_sims * tensor.tanh(
            seq_cell_with_time_sims
        )
        #
        lambda_over_seq_sims_tilde = tensor.tensordot(
            seq_hidden_with_time_sims, self.W_alpha,
            (2, 0)
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu_scale(
            lambda_over_seq_sims_tilde
        )
        # N * size_batch * dim_process
        # (2,0,1) --> dim_process * N * size_batch
        '''
        this block is to compute intensity
        '''
        self.lambda_samples = lambda_over_seq_sims.transpose((2,0,1)) * seq_sims_mask[None,:,:]
        self.num_of_samples = tensor.sum(seq_sims_mask)
        #
        #
    #
    #
    def compute_prediction_loss(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_mask,
        time_diffs
    ):
        #
        print "computing predictions loss for neural Hawkes with continuous-time LSTM ... "
        seq_emb_event = self.Emb_event[seq_type_event, :]
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        initial_cell_target_mat = tensor.outer(
            self.expand, self.c_0_target
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden_t, seq_cell_t, seq_cell_target, seq_cell, seq_decay_cell, seq_gate_output], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(
                    input=seq_emb_event[:-1, :, :],
                    taps=[0]
                ),
                dict(
                    input=seq_time_values[1:, :],
                    taps=[0]
                )
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1]),
                dict(initial=initial_cell_target_mat, taps=[-1]),
                None, None, None
            ],
            non_sequences = None
        )
        # seq_hidden_t : T * size_batch * dim_model
        seq_cell_with_time = seq_cell_target[
            :, :, :, None
        ] + (
            seq_cell[:, :, :, None] - seq_cell_target[:, :, :, None]
        ) * tensor.exp(
            -seq_decay_cell[:, :, :, None] * time_diffs[
                None, None, None, :
            ]
        )
        # T * size_batch * dim_model * M
        seq_hidden_with_time = seq_gate_output[
            :, :, :, None
        ] * tensor.tanh(
            seq_cell_with_time
        )
        # T * size_batch * dim_model * M
        lambda_over_seq_tilde = tensor.sum(
            seq_hidden_with_time[
                :, :, :, None, :
            ] * self.W_alpha[
                None, None, :, :, None
            ], axis = 2
        )
        # T * size_batch * dim_process * M
        # each time stamp, each seq in batch
        # each process, each simulation for prediction
        lambda_over_seq = self.soft_relu_scale(
            lambda_over_seq_tilde.dimshuffle(3,0,1,2)
        ).dimshuffle(1,2,3,0)
        #
        # T * size_batch * dim_process * M
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis=2
        )
        # T * size_batch * M
        term_1 = time_diffs
        # M *
        #
        cum_num = tensor.arange(
            time_diffs.shape[0]+numpy.int32(1)
        )[1:] * numpy.float32(1.0)
        # M
        term_2 = tensor.exp(
            (
                -1.0 * tensor.extra_ops.cumsum(
                    lambda_sum_over_seq, axis = 2
                ) / cum_num[None, None, :]
            ) * time_diffs[
                None, None, :
            ]
        )
        # T * size_batch * M
        term_3 = lambda_sum_over_seq
        # T * size_batch * M
        density = term_2 * term_3
        # T * size_batch * M
        time_prediction = tensor.mean(
            term_1[None, None, :] * density,
            axis = 2
        ) * time_diffs[-1]
        # T * size_batch
        lambda_over_seq_over_sims = lambda_over_seq[
            :, :, :, :
        ] * density[
            :, :, None, :
        ] / lambda_sum_over_seq[
            :, :, None, :
        ]
        # T * size_batch * dim_process * M
        prob_over_seq_over_type = tensor.mean(
            lambda_over_seq_over_sims, axis = 3
        ) * time_diffs[-1]
        # T * size_batch * dim_process
        prob_over_seq_over_type /= tensor.sum(
            prob_over_seq_over_type,
            axis=2,
            keepdims=True
        )
        # T * size_batch * dim_process
        #type_prediction = tensor.argmax(
        #    prob_over_seq_over_type, axis = 2
        #)
        # T * size_batch
        # Now we have :
        # time_prediction, type_prediction, seq_mask
        # all of -- T * size_batch
        target_type = seq_type_event[1:, :]
        target_time = seq_time_values[1:, :]
        # Type first
        new_shape_0 = target_type.shape[0] * target_type.shape[1]
        new_shape_1 = self.dim_process
        back_shape_0 = target_type.shape[0]
        back_shape_1 = target_type.shape[1]
        #
        prob_over_seq = prob_over_seq_over_type.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            target_type.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        log_prob_over_seq = tensor.log(
            prob_over_seq + numpy.float32(1e-9)
        )
        log_prob_over_seq *= seq_mask
        self.log_likelihood_type_predict = tensor.sum(
            log_prob_over_seq
        )
        #diff_type = tensor.abs_(
        #    target_type - type_prediction
        #) * seq_mask
        #diff_type = tensor.switch(
        #    diff_type >= numpy.float32(0.5),
        #    numpy.float32(1.0), numpy.float32(0.0)
        #)
        #
        #self.num_of_errors = tensor.sum(diff_type)
        # Time
        diff_time = (
            target_time - time_prediction
        )**2
        diff_time *= seq_mask
        self.square_errors = tensor.sum(diff_time)
        self.num_of_events = tensor.sum(seq_mask)
        #TODO: Hamming loss for prediction checking
        #
        type_prediction = tensor.argmax(
            prob_over_seq_over_type, axis = 2
        )
        diff_type = tensor.abs_(
            target_type - type_prediction
        ) * seq_mask
        diff_type = tensor.switch(
            diff_type >= numpy.float32(0.5),
            numpy.float32(1.0), numpy.float32(0.0)
        )
        self.num_of_errors = tensor.sum(diff_type)
        #
        self.cost_to_optimize = -self.log_likelihood_type_predict / self.num_of_events + self.square_errors / self.num_of_events + self.term_reg
        #self.cost_to_optimize = -self.log_likelihood_type_predict + self.term_reg
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        self.abs_grad_params = 0.0
        for grad_param in self.grad_params:
            self.abs_grad_params += tensor.sum(
                tensor.abs_(
                    grad_param
                )
            )
        #
        #
    #
    #
    #
    #TODO: memory efficient version of prediction loss
    def predict_each_step(
        self,
        cell_target, cell,
        decay_cell, gate_output,
        time_diffs
    ):
        # seqs : size_batch * dim_model
        # time_diffs : M
        cell_with_time = cell_target[
            :, :, None
        ] + (
            cell[:, :, None] - cell_target[:, :, None]
        ) * tensor.exp(
            -decay_cell[:, :, None] * time_diffs[
                None, None, :
            ]
        )
        # size_batch * dim_model * M
        hidden_with_time = gate_output[
            :, :, None
        ] * tensor.tanh(
            cell_with_time
        )
        # size_batch * dim_model * M
        lambda_tilde = tensor.sum(
            hidden_with_time[
                :, :, None, :
            ] * self.W_alpha[
                None, :, :, None
            ], axis = 1
        )
        # size_batch * dim_process * M
        lambda_each_step = self.soft_relu_scale(
            lambda_tilde.dimshuffle(2, 0, 1)
        ).dimshuffle(1, 2, 0)
        lambda_sum_each_step = tensor.sum(
            lambda_each_step, axis=1
        )
        # size_batch * M
        #TODO: compute integral
        term_1 = time_diffs
        cum_num = tensor.arange(
            time_diffs.shape[0]+numpy.int32(1)
        )[1:] * numpy.float32(1.0)
        # M
        term_2 = tensor.exp(
            (
                -1.0 * tensor.extra_ops.cumsum(
                    lambda_sum_each_step, axis=1
                ) / cum_num[None, :]
            ) * time_diffs[None, :]
        )
        # size_batch * M
        term_3 = lambda_sum_each_step
        density = term_2 * term_3
        # size_batch * M
        time_prediction_each_step = tensor.mean(
            term_1[None, :] * density, axis=1
        ) * time_diffs[-1]
        # size_batch
        lambda_each_step_over_sims = lambda_each_step[
            :, :, :
        ] * density[
            :, None, :
        ] / lambda_sum_each_step[
            :, None, :
        ]
        # size_batch * dim_process * M
        prob_over_type = tensor.mean(
            lambda_each_step_over_sims, axis=2
        ) * time_diffs[-1]
        # size_batch * dim_process
        prob_over_type /= tensor.sum(
            prob_over_type, axis=1, keepdims=True
        )
        # size_batch * dim_process
        return prob_over_type, time_prediction_each_step
    #
    #
    def compute_prediction_loss_lessmem(
        self,
        seq_type_event,
        seq_time_values,
        seq_mask,
        time_diffs
    ):
        #
        print "computing predictions loss of neural Hawkes with continuous-time LSTM ... "
        print "memory efficient version ... "
        seq_emb_event = self.Emb_event[seq_type_event, :]
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        initial_cell_target_mat = tensor.outer(
            self.expand, self.c_0_target
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden_t, seq_cell_t, seq_cell_target, seq_cell, seq_decay_cell, seq_gate_output], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(
                    input=seq_emb_event[:-1, :, :],
                    taps=[0]
                ),
                dict(
                    input=seq_time_values[1:, :],
                    taps=[0]
                )
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1]),
                dict(initial=initial_cell_target_mat, taps=[-1]),
                None, None, None
            ],
            non_sequences = None
        )
        #
        #TODO: predict time and type for each step
        [prob_over_seq_over_type, time_prediction], _ = theano.scan(
            fn = self.predict_each_step,
            sequences = [
                dict(input=seq_cell_target, taps=[0]),
                dict(input=seq_cell, taps=[0]),
                dict(input=seq_decay_cell, taps=[0]),
                dict(input=seq_gate_output, taps=[0])
            ],
            outputs_info = [
                None, None
            ],
            non_sequences = time_diffs
        )
        #
        target_type = seq_type_event[1:, :]
        target_time = seq_time_values[1:, :]
        # Type first
        new_shape_0 = target_type.shape[0] * target_type.shape[1]
        new_shape_1 = self.dim_process
        back_shape_0 = target_type.shape[0]
        back_shape_1 = target_type.shape[1]
        #
        prob_over_seq = prob_over_seq_over_type.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            target_type.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        log_prob_over_seq = tensor.log(
            prob_over_seq + numpy.float32(1e-9)
        )
        log_prob_over_seq *= seq_mask
        self.log_likelihood_type_predict = tensor.sum(
            log_prob_over_seq
        )
        #
        # Time
        diff_time = (
            target_time - time_prediction
        )**2
        diff_time *= seq_mask
        self.square_errors = tensor.sum(diff_time)
        self.num_of_events = tensor.sum(seq_mask)
        #TODO: Hamming loss for prediction checking
        #
        type_prediction = tensor.argmax(
            prob_over_seq_over_type, axis = 2
        )
        diff_type = tensor.abs_(
            target_type - type_prediction
        ) * seq_mask
        diff_type = tensor.switch(
            diff_type >= numpy.float32(0.5),
            numpy.float32(1.0), numpy.float32(0.0)
        )
        self.num_of_errors = tensor.sum(diff_type)
        #
        self.cost_to_optimize = -self.log_likelihood_type_predict / self.num_of_events + self.square_errors / self.num_of_events + self.term_reg
        #self.cost_to_optimize = -self.log_likelihood_type_predict + self.term_reg
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        self.abs_grad_params = 0.0
        for grad_param in self.grad_params:
            self.abs_grad_params += tensor.sum(
                tensor.abs_(
                    grad_param
                )
            )
        #
        #
    #
    #
    #
    def get_model(self):
        print "getting model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        return model_dict
        #
    #
    #
    def save_model(self, file_save):
        model_dict = self.get_model()
        print "saving model ... "
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
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
# most of them keep the decomposable structure of Hawkes
# and try to use neural networks to parametrize it
#
#
#
class HawkesInhibCTSM(object):
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        print "initializing Hawkes CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            # initialize variables
            self.mu = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='mu'
            )
            '''
            we need to notice that: in these matrices of K * K
            the (i, j) entry is the effect of j-th to i-th
            this order may be changed in the neural hawkes
            for the sake of implementation ease or the convention of Theano
            '''
            self.alpha = theano.shared(
                numpy.ones(
                    (self.dim_process, self.dim_process),
                    dtype=dtype
                ), name='alpha'
            )
            self.delta = theano.shared(
                numpy.ones(
                    (self.dim_process, self.dim_process),
                    dtype=dtype
                ), name='delta'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.mu = theano.shared(
                model_pre_train['mu'], name='mu'
            )
            self.alpha = theano.shared(
                model_pre_train['alpha'], name='alpha'
            )
            self.delta = theano.shared(
                model_pre_train['delta'], name='delta'
            )
        #
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            self.mu, self.alpha, self.delta
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        return tensor.log(numpy.float32(1.0)+tensor.exp(x))
    #
    def compute_loss(
        self,
        seq_time_to_current, seq_type_event,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask, seq_mask_to_current,
        seq_sims_time_to_current,
        seq_sims_mask_to_current,
        seq_sims_mask
    ):
        '''
        use this function to compute negative log likelihood
        seq_time_to_end : T * size_batch -- T-t_i
        seq_time_to_current : T * T * size_batch --
        for each batch, it is T * T, and at each time step t,
        it tracks the ( t_i - t_i' ) for all t_i' < t_i
        seq_type_event : T * size_batch -- for each data
        and each time step, tracks the type of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        #
        seq_mask : T * size_batch -- 1/0
        seq_mask_to_current : T * T * size_batch -- 1/0
        #
        seq_sims_mask : N * size_batch -- 1/0
        '''
        print "computing loss function of Hawkes model ... "
        # first compute the 3rd term in loss
        alpha_over_seq = self.alpha[
            :, seq_type_event
        ] # dim_process * T * size_batch
        delta_over_seq = self.delta[
            :, seq_type_event
        ] # dim_process * T * size_batch
        #
        lambda_over_seq_sims_tilde = self.mu[:,None,None] + tensor.sum(
            (
                seq_sims_mask_to_current[None,:,:,:] * (
                    alpha_over_seq[:,None,:,:] * tensor.exp(
                        -delta_over_seq[:,None,:,:] * seq_sims_time_to_current[None,:,:,:]
                    )
                )
            ), axis=2
        ) # dim_process * N * size_batch
        #
        lambda_over_seq_sims = tensor.log(
            numpy.float32(1.0) + tensor.exp(
                lambda_over_seq_sims_tilde
            )
        )
        # dim_process * N * size_batch
        #
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=0
        )
        # N * size_batch
        # mask the lambda of simulations
        lambda_sum_over_seq_sims *= seq_sims_mask
        #
        #
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        # (size_batch, )
        term_2 = numpy.float32(0.0)
        #
        '''
        for this model, the computation of term_3 follows the same procedure of term_1, since we need to estimate lambda(s_j), i.e, we need large N * T * size_batch tensors for (1) time to current; (2) mask for (1).
        then we can just follow the steps of term_1 to finish the integral estimation.
        correspondingly, we need to modify the data processors, to generate the big tensors
        '''
        # then we compute the 1st term, which is the trickest
        # we use seq_time_to_current : T * T * size_batch
        # seq_mask_to_current : T * T * size_batch
        lambda_over_seq_tilde = self.mu[:, None, None] + tensor.sum(
            (
                seq_mask_to_current[None,:,:,:]
                * (
                    alpha_over_seq[:,None,:,:] * tensor.exp(
                        -delta_over_seq[:,None,:,:]
                        * seq_time_to_current[None,:,:,:]
                    )
                )
            )
            , axis=2
        ) # dim_process * T * size_batch
        #
        lambda_over_seq = tensor.log(
            numpy.float32(1.0) + tensor.exp(
                lambda_over_seq_tilde
            )
        ) # dim_process * T * size_batch
        #
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis=0
        ) # T * size_batch
        # now we choose the right lambda for each step
        # by using seq_type_event : T * size_batch
        new_shape_0 = lambda_over_seq.shape[1]*lambda_over_seq.shape[2]
        new_shape_1 = lambda_over_seq.shape[0]
        #
        back_shape_0 = lambda_over_seq.shape[1]
        back_shape_1 = lambda_over_seq.shape[2]
        #
        lambda_target_over_seq = lambda_over_seq.transpose(
            (1,2,0)
        ).reshape(
            (
                new_shape_0, new_shape_1
            )
        )[
            tensor.arange(new_shape_0),
            seq_type_event.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
#
class NeuralHawkesCTSM(object):
    #
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Neural Hawkes CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            self.dim_model = settings['dim_model']
            # initialize variables
            self.mu = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='mu'
            )
            '''
            we need to notice that: in these matrices of D * K
            the (i, j) entry is the effect of i-th dimension
            to j-th event
            this order may be different from that of Hawkes
            so we need to be careful when interpreting
            '''
            self.delta = theano.shared(
                numpy.ones(
                    (self.dim_model, self.dim_process),
                    dtype=dtype
                ), name='delta'
            )
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            self.Emb_time = theano.shared(
                utils.sample_weights(
                    self.dim_time, self.dim_model
                ), name='Emb_time'
            )
            self.W_recur = theano.shared(
                utils.sample_weights(
                    3*self.dim_model, 4*self.dim_model
                ), name='W_recur'
            )
            self.b_recur = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.mu = theano.shared(
                model_pre_train['mu'], name='mu'
            )
            self.delta = theano.shared(
                model_pre_train['delta'], name='delta'
            )
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            self.Emb_time = theano.shared(
                model_pre_train['Emb_time'], name='Emb_time'
            )
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            self.mu, self.delta,
            self.W_alpha,
            self.Emb_event, self.Emb_time,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        return tensor.log(numpy.float32(1.0)+tensor.exp(x))
    #
    #
    def rnn_unit(
        self, emb_event_t, emb_time_t,
        hidden_tm1, cell_tm1
    ):
        pre_transform = tensor.concatenate(
            [emb_event_t, emb_time_t, hidden_tm1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        #
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[:, 3*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_tm1 + gate_input * gate_pre_c
        hidden_t = gate_output * tensor.tanh(cell_t)
        return hidden_t, cell_t
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, seq_time_rep,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
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
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        seq_sims_hidden_with_time = seq_hidden_for_sims[
            :, :, :, None
        ] * tensor.exp(
            -self.delta[
                None, None, :, :
            ] * seq_sims_time_to_current[
                :, :, None, None
            ]
        )
        #
        # N * size_batch * dim_model * dim_process
        # self.W_alpha : dim_model * dim_process
        lambda_over_seq_sims_tilde = self.mu[None, None, :] + tensor.sum(
            seq_sims_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        term_2 = numpy.float32(0.0)
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_for_lambda : T * size_batch * dim_model
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None
        ] * tensor.exp(
            -self.delta[
                None, None, :, :
            ] * seq_time_to_current[
                :, :, None, None
            ]
        )
        # T * size_batch * dim_model * dim_process
        lambda_over_seq_tilde = self.mu[None, None, :] + tensor.sum(
            seq_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # T * size_batch * dim_process
        lambda_over_seq = self.soft_relu(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        #
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
class GeneralizedNeuralHawkesCTSM(object):
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Generalized Neural Hawkes CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            self.dim_model = settings['dim_model']
            # initialize variables
            self.mu = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='mu'
            )
            '''
            we need to notice that: in these matrices of D * K
            the (i, j) entry is the effect of i-th dimension
            to j-th event
            this order may be different from that of Hawkes
            so we need to be careful when interpreting
            '''
            #self.delta = theano.shared(
            #    numpy.ones(
            #        (self.dim_model, self.dim_process),
            #        dtype=dtype
            #    ), name='delta'
            #)
            #
            self.W_delta = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model,
                            self.dim_model,
                            self.dim_process
                        )
                    )
                ), name = 'W_delta'
            )
            # the 0-th axis -- self.dim_model
            # is for dot product with hidden units
            # dot(h, W_delta) --> delta of size:
            # dim_model * dim_process
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            self.Emb_time = theano.shared(
                utils.sample_weights(
                    self.dim_time, self.dim_model
                ), name='Emb_time'
            )
            self.W_recur = theano.shared(
                utils.sample_weights(
                    3*self.dim_model, 4*self.dim_model
                ), name='W_recur'
            )
            self.b_recur = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.mu = theano.shared(
                model_pre_train['mu'], name='mu'
            )
            #self.delta = theano.shared(
            #    model_pre_train['delta'], name='delta'
            #)
            self.W_delta = theano.shared(
                model_pre_train['W_delta'], name='W_delta'
            )
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            self.Emb_time = theano.shared(
                model_pre_train['Emb_time'], name='Emb_time'
            )
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            self.mu, #self.delta,
            self.W_delta, self.W_alpha,
            self.Emb_event, self.Emb_time,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        # tensor[(x == 0).nonzeros()]
        #v_max = numpy.float32(1e9)
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        #a = tensor.switch(z>=v_max, v_max, z)
        #y[(x>=100.0).nonzeros()] = x[(x>=100.0).nonzeros()]
        #np.finfo(np.float32).max
        return z
    #
    #
    def rnn_unit(
        self, emb_event_t, emb_time_t,
        hidden_tm1, cell_tm1
    ):
        pre_transform = tensor.concatenate(
            [emb_event_t, emb_time_t, hidden_tm1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        #
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[:, 3*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_tm1 + gate_input * gate_pre_c
        hidden_t = gate_output * tensor.tanh(cell_t)
        return hidden_t, cell_t
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, seq_time_rep,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
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
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        # self.W_delta : dim_model * dim_model * dim_process
        #
        delta_for_sims = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_sims, self.W_delta, (2,0)
            )
        )
        #
        # N * size_batch * dim_model * dim_process
        #
        seq_sims_hidden_with_time = seq_hidden_for_sims[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_sims * seq_sims_time_to_current[
                :, :, None, None
            ]
        )
        #
        # N * size_batch * dim_model * dim_process
        # self.W_alpha : dim_model * dim_process
        lambda_over_seq_sims_tilde = self.mu[None, None, :] + tensor.sum(
            seq_sims_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        term_2 = numpy.float32(0.0)
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_for_lambda : T * size_batch * dim_model
        #
        #
        delta_for_lambda = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_lambda, self.W_delta, (2,0)
            )
        )
        # T * size_batch * dim_model * dim_process
        #
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_lambda * seq_time_to_current[
                :, :, None, None
            ]
        )
        # T * size_batch * dim_model * dim_process
        lambda_over_seq_tilde = self.mu[None, None, :] + tensor.sum(
            seq_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # T * size_batch * dim_process
        lambda_over_seq = self.soft_relu(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        #
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
class NeuralHawkesAdaptiveBaseCTSM(object):
    #TODO: the base rate is adaptive
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Generalized Neural Hawkes with Adaptive Base Rate CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            self.dim_model = settings['dim_model']
            # initialize variables
            #self.mu = theano.shared(
            #    numpy.ones(
            #        (self.dim_process,), dtype=dtype
            #    ), name='mu'
            #)
            '''
            we need to notice that: in these matrices of D * K
            the (i, j) entry is the effect of i-th dimension
            to j-th event
            this order may be different from that of Hawkes
            so we need to be careful when interpreting
            '''
            #self.delta = theano.shared(
            #    numpy.ones(
            #        (self.dim_model, self.dim_process),
            #        dtype=dtype
            #    ), name='delta'
            #)
            #
            self.W_mu = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model, self.dim_process
                        )
                    )
                ), name = 'W_mu'
            )
            #
            #
            self.W_delta = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model,
                            self.dim_model,
                            self.dim_process
                        )
                    )
                ), name = 'W_delta'
            )
            # the 0-th axis -- self.dim_model
            # is for dot product with hidden units
            # dot(h, W_delta) --> delta of size:
            # dim_model * dim_process
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            self.Emb_time = theano.shared(
                utils.sample_weights(
                    self.dim_time, self.dim_model
                ), name='Emb_time'
            )
            self.W_recur = theano.shared(
                utils.sample_weights(
                    3*self.dim_model, 4*self.dim_model
                ), name='W_recur'
            )
            self.b_recur = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.W_mu = theano.shared(
                model_pre_train['W_mu'], name='W_mu'
            )
            #self.delta = theano.shared(
            #    model_pre_train['delta'], name='delta'
            #)
            self.W_delta = theano.shared(
                model_pre_train['W_delta'], name='W_delta'
            )
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            self.Emb_time = theano.shared(
                model_pre_train['Emb_time'], name='Emb_time'
            )
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            #self.mu, #self.delta,
            self.W_mu, self.W_delta, self.W_alpha,
            self.Emb_event, self.Emb_time,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        return tensor.log(numpy.float32(1.0)+tensor.exp(x))
    #
    #
    def rnn_unit(
        self, emb_event_t, emb_time_t,
        hidden_tm1, cell_tm1
    ):
        pre_transform = tensor.concatenate(
            [emb_event_t, emb_time_t, hidden_tm1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        #
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[:, 3*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_tm1 + gate_input * gate_pre_c
        hidden_t = gate_output * tensor.tanh(cell_t)
        return hidden_t, cell_t
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, seq_time_rep,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
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
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        # self.W_delta : dim_model * dim_model * dim_process
        #
        delta_for_sims = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_sims, self.W_delta, (2,0)
            )
        )
        #
        # N * size_batch * dim_model * dim_process
        #
        seq_sims_hidden_with_time = seq_hidden_for_sims[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_sims * seq_sims_time_to_current[
                :, :, None, None
            ]
        )
        #
        # N * size_batch * dim_model * dim_process
        # self.W_alpha : dim_model * dim_process
        mu_for_sims = tensor.tensordot(
            seq_hidden_for_sims, self.W_mu, (2,0)
        )
        # N * size_batch * dim_process
        #
        lambda_over_seq_sims_tilde = mu_for_sims + tensor.sum(
            seq_sims_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        term_2 = numpy.float32(0.0)
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_for_lambda : T * size_batch * dim_model
        #
        #
        delta_for_lambda = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_lambda, self.W_delta, (2,0)
            )
        )
        # T * size_batch * dim_model * dim_process
        #
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_lambda * seq_time_to_current[
                :, :, None, None
            ]
        )
        # T * size_batch * dim_model * dim_process
        #
        mu_for_lambda = tensor.tensordot(
            seq_hidden_for_lambda, self.W_mu, (2,0)
        )
        # T * size_batch * dim_process
        #
        lambda_over_seq_tilde = mu_for_lambda + tensor.sum(
            seq_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # T * size_batch * dim_process
        lambda_over_seq = self.soft_relu(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        #
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
class NeuralHawkesSimpleCTSM(object):
    #TODO: all parameters controlled by one LSTM state
    #
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Generalized Neural Hawkes with Adaptive Base Rate CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            self.dim_model = settings['dim_model']
            # initialize variables
            #self.mu = theano.shared(
            #    numpy.ones(
            #        (self.dim_process,), dtype=dtype
            #    ), name='mu'
            #)
            '''
            we need to notice that: in these matrices of D * K
            the (i, j) entry is the effect of i-th dimension
            to j-th event
            this order may be different from that of Hawkes
            so we need to be careful when interpreting
            '''
            #self.delta = theano.shared(
            #    numpy.ones(
            #        (self.dim_model, self.dim_process),
            #        dtype=dtype
            #    ), name='delta'
            #)
            #
            self.W_hawkes = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model,
                            3 * self.dim_process
                        )
                    )
                ), name = 'W_hawkes'
            )
            self.b_hawkes = theano.shared(
                numpy.zeros(
                    (3*self.dim_process,), dtype=dtype
                ), name='b_hawkes'
            )
            #
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            self.Emb_time = theano.shared(
                utils.sample_weights(
                    self.dim_time, self.dim_model
                ), name='Emb_time'
            )
            self.W_recur = theano.shared(
                utils.sample_weights(
                    3*self.dim_model, 4*self.dim_model
                ), name='W_recur'
            )
            self.b_recur = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.W_hawkes = theano.shared(
                model_pre_train['W_hawkes'], name = 'W_hawkes'
            )
            self.b_hawkes = theano.shared(
                model_pre_train['b_hawkes'], name='b_hawkes'
            )
            #
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            self.Emb_time = theano.shared(
                model_pre_train['Emb_time'], name='Emb_time'
            )
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            #self.mu, #self.delta,
            self.W_hawkes, self.b_hawkes,
            self.Emb_event, self.Emb_time,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        return tensor.log(numpy.float32(1.0)+tensor.exp(x))
    #
    #
    def rnn_unit(
        self, emb_event_t, emb_time_t,
        hidden_tm1, cell_tm1
    ):
        pre_transform = tensor.concatenate(
            [emb_event_t, emb_time_t, hidden_tm1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        #
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[:, 3*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_tm1 + gate_input * gate_pre_c
        hidden_t = gate_output * tensor.tanh(cell_t)
        return hidden_t, cell_t
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, seq_time_rep,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
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
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        # self.W_delta : dim_model * dim_model * dim_process
        #
        params_hawkes_for_sims = tensor.tensordot(
            seq_hidden_for_sims, self.W_hawkes, (2,0)
        ) + self.b_hawkes[None, None, :]
        #
        mu_for_sims = params_hawkes_for_sims[
            :, :, :self.dim_process
        ]
        alpha_for_sims = params_hawkes_for_sims[
            :, :, self.dim_process:2*self.dim_process
        ]
        delta_for_sims = self.soft_relu(
            params_hawkes_for_sims[
                :, :, 2*self.dim_process:
            ]
        )
        #
        lambda_over_seq_sims_tilde = mu_for_sims + tensor.exp(
            -delta_for_sims * seq_sims_time_to_current[
                :, :, None
            ]
        ) * alpha_for_sims
        #
        # N * size_batch * dim_process
        # may over flow here
        lambda_over_seq_sims = self.soft_relu(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        term_2 = numpy.float32(0.0)
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_for_lambda : T * size_batch * dim_model
        #
        params_hawkes_for_lambda = tensor.tensordot(
            seq_hidden_for_lambda, self.W_hawkes, (2,0)
        ) + self.b_hawkes[None, None, :]
        #
        mu_for_lambda = params_hawkes_for_lambda[
            :, :, :self.dim_process
        ]
        alpha_for_lambda = params_hawkes_for_lambda[
            :, :, self.dim_process:2*self.dim_process
        ]
        delta_for_lambda = self.soft_relu(
            params_hawkes_for_lambda[
                :, :, 2*self.dim_process:
            ]
        )
        #
        lambda_over_seq_tilde = mu_for_lambda + tensor.exp(
            -delta_for_lambda * seq_time_to_current[
                :, :, None
            ]
        ) * alpha_for_lambda
        #
        # T * size_batch * dim_process
        #
        lambda_over_seq = self.soft_relu(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        #
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
#
class NeuralHawkesCTSM_time(object):
    '''
    this model stems from neural hawkes
    but encode time (positive real values) with neural nodes
    '''
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Neural Hawkes CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            self.dim_model = settings['dim_model']
            # initialize variables
            self.mu = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='mu'
            )
            '''
            we need to notice that: in these matrices of D * K
            the (i, j) entry is the effect of i-th dimension
            to j-th event
            this order may be different from that of Hawkes
            so we need to be careful when interpreting
            '''
            self.delta = theano.shared(
                numpy.ones(
                    (self.dim_model, self.dim_process),
                    dtype=dtype
                ), name='delta'
            )
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            self.Emb_time = theano.shared(
                utils.sample_weights(
                    self.dim_time+numpy.int32(1), self.dim_model
                ), name='Emb_time'
            )
            # a dim_time vector for thresholding time
            self.Threshold_time = theano.shared(
                numpy.float32(settings['threshold_time']),
                name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                utils.sample_weights(
                    3*self.dim_model, 4*self.dim_model
                ), name='W_recur'
            )
            self.b_recur = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.mu = theano.shared(
                model_pre_train['mu'], name='mu'
            )
            self.delta = theano.shared(
                model_pre_train['delta'], name='delta'
            )
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            self.Emb_time = theano.shared(
                model_pre_train['Emb_time'], name='Emb_time'
            )
            #
            self.Threshold_time = theano.shared(
                model_pre_train['Threshold_time'], name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            self.mu, self.delta,
            self.W_alpha,
            self.Emb_event, self.Emb_time, self.Threshold_time,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        #a = tensor.switch(z>=v_max, v_max, z)
        #y[(x>=100.0).nonzeros()] = x[(x>=100.0).nonzeros()]
        #np.finfo(np.float32).max
        return z
    #
    #
    def rnn_unit(
        self, emb_event_t, emb_time_t,
        hidden_tm1, cell_tm1
    ):
        pre_transform = tensor.concatenate(
            [emb_event_t, emb_time_t, hidden_tm1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        #
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[:, 3*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_tm1 + gate_input * gate_pre_c
        hidden_t = gate_output * tensor.tanh(cell_t)
        return hidden_t, cell_t
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event,
        #seq_time_rep,
        seq_time_values,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        seq_time_values : (T+1) * size_batch -- for each data and each time step, track the time values of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        '''
        pass time values through thresholds
        '''
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        seq_sims_hidden_with_time = seq_hidden_for_sims[
            :, :, :, None
        ] * tensor.exp(
            -self.delta[
                None, None, :, :
            ] * seq_sims_time_to_current[
                :, :, None, None
            ]
        )
        #
        # N * size_batch * dim_model * dim_process
        # self.W_alpha : dim_model * dim_process
        lambda_over_seq_sims_tilde = self.mu[None, None, :] + tensor.sum(
            seq_sims_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        term_2 = numpy.float32(0.0)
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_for_lambda : T * size_batch * dim_model
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None
        ] * tensor.exp(
            -self.delta[
                None, None, :, :
            ] * seq_time_to_current[
                :, :, None, None
            ]
        )
        # T * size_batch * dim_model * dim_process
        lambda_over_seq_tilde = self.mu[None, None, :] + tensor.sum(
            seq_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # T * size_batch * dim_process
        lambda_over_seq = self.soft_relu(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        #
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
class GeneralizedNeuralHawkesCTSM_time(object):
    #
    '''
    this model stems from generalized neural hawkes
    but encode time (positive real values) with neural nodes
    '''
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Generalized Neural Hawkes CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            self.dim_model = settings['dim_model']
            # initialize variables
            self.mu = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='mu'
            )
            '''
            we need to notice that: in these matrices of D * K
            the (i, j) entry is the effect of i-th dimension
            to j-th event
            this order may be different from that of Hawkes
            so we need to be careful when interpreting
            '''
            #self.delta = theano.shared(
            #    numpy.ones(
            #        (self.dim_model, self.dim_process),
            #        dtype=dtype
            #    ), name='delta'
            #)
            #
            self.W_delta = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model,
                            self.dim_model,
                            self.dim_process
                        )
                    )
                ), name = 'W_delta'
            )
            # the 0-th axis -- self.dim_model
            # is for dot product with hidden units
            # dot(h, W_delta) --> delta of size:
            # dim_model * dim_process
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            self.Emb_time = theano.shared(
                utils.sample_weights(
                    self.dim_time+numpy.int32(1), self.dim_model
                ), name='Emb_time'
            )
            # a dim_time vector for thresholding time
            self.Threshold_time = theano.shared(
                numpy.float32(settings['threshold_time']),
                name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                utils.sample_weights(
                    3*self.dim_model, 4*self.dim_model
                ), name='W_recur'
            )
            self.b_recur = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.mu = theano.shared(
                model_pre_train['mu'], name='mu'
            )
            #self.delta = theano.shared(
            #    model_pre_train['delta'], name='delta'
            #)
            self.W_delta = theano.shared(
                model_pre_train['W_delta'], name='W_delta'
            )
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            self.Emb_time = theano.shared(
                model_pre_train['Emb_time'], name='Emb_time'
            )
            #
            self.Threshold_time = theano.shared(
                model_pre_train['Threshold_time'], name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            self.mu, #self.delta,
            self.W_delta, self.W_alpha,
            self.Emb_event, self.Emb_time,
            self.Threshold_time,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        # tensor[(x == 0).nonzeros()]
        #v_max = numpy.float32(1e9)
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        #a = tensor.switch(z>=v_max, v_max, z)
        #y[(x>=100.0).nonzeros()] = x[(x>=100.0).nonzeros()]
        #np.finfo(np.float32).max
        return z
    #
    #
    def rnn_unit(
        self, emb_event_t, emb_time_t,
        hidden_tm1, cell_tm1
    ):
        pre_transform = tensor.concatenate(
            [emb_event_t, emb_time_t, hidden_tm1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        #
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[:, 3*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_tm1 + gate_input * gate_pre_c
        hidden_t = gate_output * tensor.tanh(cell_t)
        return hidden_t, cell_t
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
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
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        '''
        pass time values through thresholds
        '''
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        # self.W_delta : dim_model * dim_model * dim_process
        #
        delta_for_sims = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_sims, self.W_delta, (2,0)
            )
        )
        #
        # N * size_batch * dim_model * dim_process
        #
        seq_sims_hidden_with_time = seq_hidden_for_sims[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_sims * seq_sims_time_to_current[
                :, :, None, None
            ]
        )
        #
        # N * size_batch * dim_model * dim_process
        # self.W_alpha : dim_model * dim_process
        lambda_over_seq_sims_tilde = self.mu[None, None, :] + tensor.sum(
            seq_sims_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        term_2 = numpy.float32(0.0)
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_for_lambda : T * size_batch * dim_model
        #
        #
        delta_for_lambda = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_lambda, self.W_delta, (2,0)
            )
        )
        # T * size_batch * dim_model * dim_process
        #
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_lambda * seq_time_to_current[
                :, :, None, None
            ]
        )
        # T * size_batch * dim_model * dim_process
        lambda_over_seq_tilde = self.mu[None, None, :] + tensor.sum(
            seq_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # T * size_batch * dim_process
        lambda_over_seq = self.soft_relu(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        #
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
class NeuralHawkesAdaptiveBaseCTSM_time(object):
    #TODO: the base rate is adaptive
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Generalized Neural Hawkes with Adaptive Base Rate CTSM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            self.dim_model = settings['dim_model']
            # initialize variables
            #self.mu = theano.shared(
            #    numpy.ones(
            #        (self.dim_process,), dtype=dtype
            #    ), name='mu'
            #)
            '''
            we need to notice that: in these matrices of D * K
            the (i, j) entry is the effect of i-th dimension
            to j-th event
            this order may be different from that of Hawkes
            so we need to be careful when interpreting
            '''
            #self.delta = theano.shared(
            #    numpy.ones(
            #        (self.dim_model, self.dim_process),
            #        dtype=dtype
            #    ), name='delta'
            #)
            #
            self.W_mu = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model, self.dim_process
                        )
                    )
                ), name = 'W_mu'
            )
            #
            #
            self.W_delta = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model,
                            self.dim_model,
                            self.dim_process
                        )
                    )
                ), name = 'W_delta'
            )
            # the 0-th axis -- self.dim_model
            # is for dot product with hidden units
            # dot(h, W_delta) --> delta of size:
            # dim_model * dim_process
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            self.Emb_time = theano.shared(
                utils.sample_weights(
                    self.dim_time+numpy.int32(1), self.dim_model
                ), name='Emb_time'
            )
            # a dim_time vector for thresholding time
            self.Threshold_time = theano.shared(
                numpy.float32(settings['threshold_time']),
                name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                utils.sample_weights(
                    3*self.dim_model, 4*self.dim_model
                ), name='W_recur'
            )
            self.b_recur = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.W_mu = theano.shared(
                model_pre_train['W_mu'], name='W_mu'
            )
            #self.delta = theano.shared(
            #    model_pre_train['delta'], name='delta'
            #)
            self.W_delta = theano.shared(
                model_pre_train['W_delta'], name='W_delta'
            )
            #print "W_delta is : "
            #print model_pre_train['W_delta']
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            self.Emb_time = theano.shared(
                model_pre_train['Emb_time'], name='Emb_time'
            )
            #
            self.Threshold_time = theano.shared(
                model_pre_train['Threshold_time'], name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            #self.mu, #self.delta,
            self.W_mu, self.W_delta, self.W_alpha,
            self.Emb_event, self.Emb_time,
            self.Threshold_time,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        # tensor[(x == 0).nonzeros()]
        #v_max = numpy.float32(1e9)
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        #a = tensor.switch(z>=v_max, v_max, z)
        #y[(x>=100.0).nonzeros()] = x[(x>=100.0).nonzeros()]
        #np.finfo(np.float32).max
        return z
    #
    #
    def rnn_unit(
        self, emb_event_t, emb_time_t,
        hidden_tm1, cell_tm1
    ):
        pre_transform = tensor.concatenate(
            [emb_event_t, emb_time_t, hidden_tm1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        #
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[:, 3*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_tm1 + gate_input * gate_pre_c
        hidden_t = gate_output * tensor.tanh(cell_t)
        return hidden_t, cell_t
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
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
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        '''
        pass time values through thresholds
        '''
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        # self.W_delta : dim_model * dim_model * dim_process
        #
        delta_for_sims = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_sims, self.W_delta, (2,0)
            )
        )
        #
        # N * size_batch * dim_model * dim_process
        #
        seq_sims_hidden_with_time = seq_hidden_for_sims[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_sims * seq_sims_time_to_current[
                :, :, None, None
            ]
        )
        #
        # N * size_batch * dim_model * dim_process
        # self.W_alpha : dim_model * dim_process
        mu_for_sims = tensor.tensordot(
            seq_hidden_for_sims, self.W_mu, (2,0)
        )
        # N * size_batch * dim_process
        #
        lambda_over_seq_sims_tilde = mu_for_sims + tensor.sum(
            seq_sims_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        term_2 = numpy.float32(0.0)
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_for_lambda : T * size_batch * dim_model
        #
        #
        delta_for_lambda = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_lambda, self.W_delta, (2,0)
            )
        )
        # T * size_batch * dim_model * dim_process
        #
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_lambda * seq_time_to_current[
                :, :, None, None
            ]
        )
        # T * size_batch * dim_model * dim_process
        #
        mu_for_lambda = tensor.tensordot(
            seq_hidden_for_lambda, self.W_mu, (2,0)
        )
        # T * size_batch * dim_process
        #
        lambda_over_seq_tilde = mu_for_lambda + tensor.sum(
            seq_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # T * size_batch * dim_process
        lambda_over_seq = self.soft_relu(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        #
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def compute_prediction(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_mask,
        time_diffs
    ):
        '''
        use this function to compute log likelihood
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_values : (T+1) * size_batch -- t_i - t_i-1
        seq_mask : T * size_batch -- 1/0
        time_diffs : vector of some length
        sample diff time for each item in each batch
        same within one batch
        '''
        print "computing predictions ... "
        seq_emb_event = self.Emb_event[seq_type_event, :]
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        # seq_hidden : (T+1) * size_batch * dim_model
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # seq_hidden_for_lambda :
        # T * size_batch * dim_model
        delta_for_lambda_pre = tensor.tensordot(
            seq_hidden_for_lambda, self.W_delta, (2,0)
        )
        #
        delta_for_lambda = self.soft_relu(
            delta_for_lambda_pre
        )
        # T * size_batch * dim_model * dim_process
        # time_diffs : M * --> a vector
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None, None
        ] * tensor.exp(
            -delta_for_lambda[
                :, :, :, :, None
            ] * time_diffs[
                None, None, None, None, :
            ]
        )
        # T * size_batch * dim_model * dim_process * M
        mu_for_lambda = tensor.tensordot(
            seq_hidden_for_lambda, self.W_mu, (2,0)
        )
        # T * size_batch * dim_process
        lambda_over_seq_tilde = tensor.sum(
            seq_hidden_with_time * self.W_alpha[
                None, None, :, :, None
            ], axis = 2
        ) + mu_for_lambda[:, :, :, None]
        # T * size_batch * dim_process * M
        # each time stamp, each seq in batch
        # each process, each simulation for prediction
        lambda_over_seq = self.soft_relu(
            lambda_over_seq_tilde
        )
        #
        # T * size_batch * dim_process * M
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis=2
        )
        # T * size_batch * M
        cum_num = tensor.arange(
            time_diffs.shape[0]+numpy.int32(1)
        )[1:] * numpy.float32(1.0)
        # M
        term_1 = time_diffs
        # M *
        term_2 = tensor.exp(
            -1.0 * tensor.extra_ops.cumsum(
                lambda_sum_over_seq, axis = 2
            ) / cum_num[None, None, :]
        )
        # T * size_batch * M
        term_3 = lambda_sum_over_seq
        # T * size_batch * M
        density = term_2 * term_3
        time_prediction = tensor.mean(
            term_1[None, None, :] * density,
            axis = 2
        )
        # T * size_batch
        lambda_over_seq_over_sims = lambda_over_seq[
            :, :, :, :
        ] * density[
            :, :, None, :
        ] / lambda_sum_over_seq[
            :, :, None, :
        ]
        # T * size_batch * dim_process * M
        prob_over_seq_over_type = tensor.mean(
            lambda_over_seq_over_sims, axis = 3
        )
        # T * size_batch * dim_process
        type_prediction = tensor.argmax(
            prob_over_seq_over_type, axis = 2
        )
        # T * size_batch
        # Now we have :
        # time_prediction, type_prediction, seq_mask
        # all of -- T * size_batch
        target_type = seq_type_event[1:, :]
        target_time = seq_time_values[1:, :]
        # Type first
        diff_type = tensor.abs_(
            target_type - type_prediction
        ) * seq_mask
        diff_type = tensor.switch(
            diff_type >= numpy.float32(0.5),
            numpy.float32(1.0), numpy.float32(0.0)
        )
        self.num_of_errors = tensor.sum(diff_type)
        # Time
        diff_time = (
            target_time - time_prediction
        )**2
        diff_time *= seq_mask
        self.square_errors = tensor.sum(diff_time)
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        #
        #TODO: for debug
        #self.time_prediction = time_prediction
        #self.target_time = target_time
        #self.type_prediction = type_prediction
        #self.target_type = target_type
        #
        #self.seq_hidden = seq_hidden_for_lambda[-1,0,:]
        #self.intensity = lambda_over_seq
        #self.cum_num = cum_num
        #self.density = density
        #self.seq_delta_pre = delta_for_lambda_pre[-1,0,:]
        #self.seq_delta_pre_check = tensor.dot(
        #    self.seq_hidden, self.W_delta
        #)
        #self.seq_delta = delta_for_lambda
        #self.lambda_tilde = lambda_over_seq_tilde
        #
    #
    #
    def get_model(self):
        print "getting model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        return model_dict
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
#
class NeuralHawkesAdaptiveBaseCTSM_time_scale(object):
    #TODO: the base rate is adaptive
    # and it uses neural time encoder
    # and it uses scale parameter  s_k
    # to addjust the soft_relu 's curvature
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Generalized Neural Hawkes with Adaptive Base Rate CTSM with neural time encoder and scale s_k ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            self.dim_model = settings['dim_model']
            # initialize variables
            #self.mu = theano.shared(
            #    numpy.ones(
            #        (self.dim_process,), dtype=dtype
            #    ), name='mu'
            #)
            '''
            we need to notice that: in these matrices of D * K
            the (i, j) entry is the effect of i-th dimension
            to j-th event
            this order may be different from that of Hawkes
            so we need to be careful when interpreting
            '''
            #self.delta = theano.shared(
            #    numpy.ones(
            #        (self.dim_model, self.dim_process),
            #        dtype=dtype
            #    ), name='delta'
            #)
            self.scale = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='scale'
            )
            #
            self.W_mu = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model, self.dim_process
                        )
                    )
                ), name = 'W_mu'
            )
            #
            #
            self.W_delta = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model,
                            self.dim_model,
                            self.dim_process
                        )
                    )
                ), name = 'W_delta'
            )
            # the 0-th axis -- self.dim_model
            # is for dot product with hidden units
            # dot(h, W_delta) --> delta of size:
            # dim_model * dim_process
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            self.Emb_time = theano.shared(
                utils.sample_weights(
                    self.dim_time+numpy.int32(1), self.dim_model
                ), name='Emb_time'
            )
            # a dim_time vector for thresholding time
            self.Threshold_time = theano.shared(
                numpy.float32(settings['threshold_time']),
                name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                utils.sample_weights(
                    3*self.dim_model, 4*self.dim_model
                ), name='W_recur'
            )
            self.b_recur = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.scale = theano.shared(
                model_pre_train['scale'], name='scale'
            )
            #
            self.W_mu = theano.shared(
                model_pre_train['W_mu'], name='W_mu'
            )
            #self.delta = theano.shared(
            #    model_pre_train['delta'], name='delta'
            #)
            #
            self.W_delta = theano.shared(
                model_pre_train['W_delta'], name='W_delta'
            )
            #print "W_delta is : "
            #print model_pre_train['W_delta']
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            self.Emb_time = theano.shared(
                model_pre_train['Emb_time'], name='Emb_time'
            )
            #
            self.Threshold_time = theano.shared(
                model_pre_train['Threshold_time'], name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            #self.mu, #self.delta,
            self.scale, # scale parameter
            self.W_mu, self.W_delta, self.W_alpha,
            self.Emb_event, self.Emb_time,
            self.Threshold_time,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        # tensor[(x == 0).nonzeros()]
        #v_max = numpy.float32(1e9)
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        #a = tensor.switch(z>=v_max, v_max, z)
        #y[(x>=100.0).nonzeros()] = x[(x>=100.0).nonzeros()]
        #np.finfo(np.float32).max
        return z
    #
    #
    def soft_relu_scale(self, x):
        # x is symbolic tensor
        # last dim is dim_process
        # this is important !
        x /= self.scale
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        z *= self.scale
        return z
    #
    #
    def rnn_unit(
        self, emb_event_t, emb_time_t,
        hidden_tm1, cell_tm1
    ):
        pre_transform = tensor.concatenate(
            [emb_event_t, emb_time_t, hidden_tm1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        #
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[:, 3*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_tm1 + gate_input * gate_pre_c
        hidden_t = gate_output * tensor.tanh(cell_t)
        return hidden_t, cell_t
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
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
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        '''
        pass time values through thresholds
        '''
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        # self.W_delta : dim_model * dim_model * dim_process
        #
        delta_for_sims = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_sims, self.W_delta, (2,0)
            )
        )
        #
        # N * size_batch * dim_model * dim_process
        #
        seq_sims_hidden_with_time = seq_hidden_for_sims[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_sims * seq_sims_time_to_current[
                :, :, None, None
            ]
        )
        #
        # N * size_batch * dim_model * dim_process
        # self.W_alpha : dim_model * dim_process
        mu_for_sims = tensor.tensordot(
            seq_hidden_for_sims, self.W_mu, (2,0)
        )
        # N * size_batch * dim_process
        #
        lambda_over_seq_sims_tilde = mu_for_sims + tensor.sum(
            seq_sims_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu_scale(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        term_2 = numpy.float32(0.0)
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_for_lambda : T * size_batch * dim_model
        #
        #
        delta_for_lambda = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_lambda, self.W_delta, (2,0)
            )
        )
        # T * size_batch * dim_model * dim_process
        #
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_lambda * seq_time_to_current[
                :, :, None, None
            ]
        )
        # T * size_batch * dim_model * dim_process
        #
        mu_for_lambda = tensor.tensordot(
            seq_hidden_for_lambda, self.W_mu, (2,0)
        )
        # T * size_batch * dim_process
        #
        lambda_over_seq_tilde = mu_for_lambda + tensor.sum(
            seq_hidden_with_time * self.W_alpha[
                None, None, :, :
            ],
            axis = 2
        )
        # T * size_batch * dim_process
        lambda_over_seq = self.soft_relu_scale(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        #
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    #
    def compute_prediction_loss(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_mask,
        time_diffs
    ):
        #
        print "computing predictions loss ... "
        seq_emb_event = self.Emb_event[seq_type_event, :]
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        # seq_hidden : (T+1) * size_batch * dim_model
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # seq_hidden_for_lambda :
        # T * size_batch * dim_model
        delta_for_lambda_pre = tensor.tensordot(
            seq_hidden_for_lambda, self.W_delta, (2,0)
        )
        #
        delta_for_lambda = self.soft_relu(
            delta_for_lambda_pre
        )
        # T * size_batch * dim_model * dim_process
        # time_diffs : M * --> a vector
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None, None
        ] * tensor.exp(
            -delta_for_lambda[
                :, :, :, :, None
            ] * time_diffs[
                None, None, None, None, :
            ]
        )
        # T * size_batch * dim_model * dim_process * M
        mu_for_lambda = tensor.tensordot(
            seq_hidden_for_lambda, self.W_mu, (2,0)
        )
        # T * size_batch * dim_process
        lambda_over_seq_tilde = tensor.sum(
            seq_hidden_with_time * self.W_alpha[
                None, None, :, :, None
            ], axis = 2
        ) + mu_for_lambda[:, :, :, None]
        # T * size_batch * dim_process * M
        # each time stamp, each seq in batch
        # each process, each simulation for prediction
        lambda_over_seq = self.soft_relu_scale(
            lambda_over_seq_tilde.dimshuffle(3,0,1,2)
        ).dimshuffle(1,2,3,0)
        #
        # T * size_batch * dim_process * M
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis=2
        )
        # T * size_batch * M
        term_1 = time_diffs
        # M *
        #
        #
        cum_num = tensor.arange(
            time_diffs.shape[0]+numpy.int32(1)
        )[1:] * numpy.float32(1.0)
        # M
        term_2 = tensor.exp(
            (
                -1.0 * tensor.extra_ops.cumsum(
                    lambda_sum_over_seq, axis = 2
                ) / cum_num[None, None, :]
            ) * time_diffs[
                None, None, :
            ]
        )
        #
        #term_2 = tensor.exp(
        #    -1.0 * lambda_sum_over_seq * time_diffs[
        #        None, None, :
        #    ]
        #)
        # T * size_batch * M
        term_3 = lambda_sum_over_seq
        # T * size_batch * M
        density = term_2 * term_3
        # T * size_batch * M
        time_prediction = tensor.mean(
            term_1[None, None, :] * density,
            axis = 2
        ) * time_diffs[-1]
        # T * size_batch
        lambda_over_seq_over_sims = lambda_over_seq[
            :, :, :, :
        ] * density[
            :, :, None, :
        ] / lambda_sum_over_seq[
            :, :, None, :
        ]
        # T * size_batch * dim_process * M
        prob_over_seq_over_type = tensor.mean(
            lambda_over_seq_over_sims, axis = 3
        ) * time_diffs[-1]
        # T * size_batch * dim_process
        prob_over_seq_over_type /= tensor.sum(
            prob_over_seq_over_type,
            axis=2,
            keepdims=True
        )
        # T * size_batch * dim_process
        #type_prediction = tensor.argmax(
        #    prob_over_seq_over_type, axis = 2
        #)
        # T * size_batch
        # Now we have :
        # time_prediction, type_prediction, seq_mask
        # all of -- T * size_batch
        target_type = seq_type_event[1:, :]
        target_time = seq_time_values[1:, :]
        # Type first
        new_shape_0 = target_type.shape[0] * target_type.shape[1]
        new_shape_1 = self.dim_process
        back_shape_0 = target_type.shape[0]
        back_shape_1 = target_type.shape[1]
        #
        prob_over_seq = prob_over_seq_over_type.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            target_type.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        log_prob_over_seq = tensor.log(
            prob_over_seq + numpy.float32(1e-9)
        )
        log_prob_over_seq *= seq_mask
        self.log_likelihood_type_predict = tensor.sum(
            log_prob_over_seq
        )
        #diff_type = tensor.abs_(
        #    target_type - type_prediction
        #) * seq_mask
        #diff_type = tensor.switch(
        #    diff_type >= numpy.float32(0.5),
        #    numpy.float32(1.0), numpy.float32(0.0)
        #)
        #
        #self.num_of_errors = tensor.sum(diff_type)
        # Time
        diff_time = (
            target_time - time_prediction
        )**2
        diff_time *= seq_mask
        self.square_errors = tensor.sum(diff_time)
        self.num_of_events = tensor.sum(seq_mask)
        #TODO: Hamming loss for prediction checking
        #
        type_prediction = tensor.argmax(
            prob_over_seq_over_type, axis = 2
        )
        diff_type = tensor.abs_(
            target_type - type_prediction
        ) * seq_mask
        diff_type = tensor.switch(
            diff_type >= numpy.float32(0.5),
            numpy.float32(1.0), numpy.float32(0.0)
        )
        self.num_of_errors = tensor.sum(diff_type)
        #
        self.cost_to_optimize = -self.log_likelihood_type_predict / self.num_of_events + self.square_errors / self.num_of_events + self.term_reg
        #self.cost_to_optimize = -self.log_likelihood_type_predict + self.term_reg
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        self.abs_grad_params = 0.0
        for grad_param in self.grad_params:
            self.abs_grad_params += tensor.sum(
                tensor.abs_(
                    grad_param
                )
            )
        #
        #
    #
    #TODO: memory efficient version of prediction loss
    def predict_each_step(
        self, hidden_for_lambda, time_diffs
    ):
        # hidden_for_lambda : size_batch * dim_model
        # time_diffs : M
        delta_for_lambda = self.soft_relu(
            tensor.tensordot(
                hidden_for_lambda, self.W_delta, (1,0)
            )
        )
        # delta_for_lambda : size_batch * dim_model * dim_process
        hidden_with_time = hidden_for_lambda[
            :, :, None, None
        ] * tensor.exp(
            -delta_for_lambda[
                :, :, :, None
            ] * time_diffs[
                None, None, None, :
            ]
        )
        # hidden_with_time : size_batch * dim_model * dim_process * M
        mu_for_lambda = tensor.tensordot(
            hidden_for_lambda, self.W_mu, (1,0)
        )
        # mu_for_lambda : size_batch * dim_process
        lambda_tilde = tensor.sum(
            hidden_with_time * self.W_alpha[
                None, :, : , None
            ], axis = 1
        ) + mu_for_lambda[:, :, None]
        # size_batch * dim_process * M
        lambda_each_step = self.soft_relu_scale(
            lambda_tilde.dimshuffle(2,0,1)
        ).dimshuffle(1,2,0)
        # size_batch * dim_process * M
        lambda_sum_each_step = tensor.sum(
            lambda_each_step, axis=1
        )
        # size_batch * M
        #TODO: compute integral
        term_1 = time_diffs
        cum_num = tensor.arange(
            time_diffs.shape[0]+numpy.int32(1)
        )[1:] * numpy.float32(1.0)
        # M
        term_2 = tensor.exp(
            (
                -1.0 * tensor.extra_ops.cumsum(
                    lambda_sum_each_step, axis=1
                ) / cum_num[None, :]
            ) * time_diffs[None, :]
        )
        # size_batch * M
        term_3 = lambda_sum_each_step
        density = term_2 * term_3
        # size_batch * M
        time_prediction_each_step = tensor.mean(
            term_1[None, :] * density, axis=1
        ) * time_diffs[-1]
        # size_batch
        lambda_each_step_over_sims = lambda_each_step[
            :, :, :
        ] * density[
            :, None, :
        ] / lambda_sum_each_step[
            :, None, :
        ]
        # size_batch * dim_process * M
        prob_over_type = tensor.mean(
            lambda_each_step_over_sims, axis=2
        ) * time_diffs[-1]
        # size_batch * dim_process
        prob_over_type /= tensor.sum(
            prob_over_type, axis=1, keepdims=True
        )
        # size_batch * dim_process
        return prob_over_type, time_prediction_each_step

    #
    def compute_prediction_loss_lessmem(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_mask,
        time_diffs
    ):
        #
        print "computing predictions loss ... "
        print "memory efficient version ... "
        seq_emb_event = self.Emb_event[seq_type_event, :]
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #TODO: get sequence of hidden units
        # seq_hidden : (T+1) * size_batch * dim_model
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # seq_hidden_for_lambda :
        # T * size_batch * dim_model
        #TODO: predict time and type for each step
        [prob_over_seq_over_type, time_prediction], _ = theano.scan(
            fn = self.predict_each_step,
            sequences = dict(
                input=seq_hidden_for_lambda, taps=[0]
            ),
            outputs_info = [
                None, None
            ],
            non_sequences = time_diffs
        )
        #
        target_type = seq_type_event[1:, :]
        target_time = seq_time_values[1:, :]
        # Type first
        new_shape_0 = target_type.shape[0] * target_type.shape[1]
        new_shape_1 = self.dim_process
        back_shape_0 = target_type.shape[0]
        back_shape_1 = target_type.shape[1]
        #
        prob_over_seq = prob_over_seq_over_type.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            target_type.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        log_prob_over_seq = tensor.log(
            prob_over_seq + numpy.float32(1e-9)
        )
        log_prob_over_seq *= seq_mask
        self.log_likelihood_type_predict = tensor.sum(
            log_prob_over_seq
        )
        #
        # Time
        diff_time = (
            target_time - time_prediction
        )**2
        diff_time *= seq_mask
        self.square_errors = tensor.sum(diff_time)
        self.num_of_events = tensor.sum(seq_mask)
        #TODO: Hamming loss for prediction checking
        #
        type_prediction = tensor.argmax(
            prob_over_seq_over_type, axis = 2
        )
        diff_type = tensor.abs_(
            target_type - type_prediction
        ) * seq_mask
        diff_type = tensor.switch(
            diff_type >= numpy.float32(0.5),
            numpy.float32(1.0), numpy.float32(0.0)
        )
        self.num_of_errors = tensor.sum(diff_type)
        #
        self.cost_to_optimize = -self.log_likelihood_type_predict / self.num_of_events + self.square_errors / self.num_of_events + self.term_reg
        #self.cost_to_optimize = -self.log_likelihood_type_predict + self.term_reg
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        self.abs_grad_params = 0.0
        for grad_param in self.grad_params:
            self.abs_grad_params += tensor.sum(
                tensor.abs_(
                    grad_param
                )
            )
        #
        #
    #
    #
    #
    def get_model(self):
        print "getting model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        return model_dict
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
#
class NeuralHawkesAdaptiveBaseCTSM_time_scale_r(object):
    #TODO: the base rate is adaptive
    # and it uses neural time encoder
    # and it uses scale parameter  s_k
    # to addjust the soft_relu 's curvature
    #
    # r means reduced version :
    # delta param is D * D, not D * D * K
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        #
        #
        print "initializing Generalized Neural Hawkes with Adaptive Base Rate CTSM with neural time encoder and scale s_k ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            self.dim_model = settings['dim_model']
            # initialize variables
            self.scale = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='scale'
            )
            #
            self.W_mu = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model, self.dim_process
                        )
                    )
                ), name = 'W_mu'
            )
            #
            self.W_delta = theano.shared(
                numpy.float32(
                    numpy.random.normal(
                        loc = 0.0, scale = 0.1,
                        size = (
                            self.dim_model,
                            self.dim_model
                        )
                    )
                ), name = 'W_delta'
            )
            #
            # the 0-th axis -- self.dim_model
            # is for dot product with hidden units
            # dot(h, W_delta) --> delta of size:
            # dim_model * dim_process
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            self.Emb_time = theano.shared(
                utils.sample_weights(
                    self.dim_time+numpy.int32(1), self.dim_model
                ), name='Emb_time'
            )
            # a dim_time vector for thresholding time
            self.Threshold_time = theano.shared(
                numpy.float32(settings['threshold_time']),
                name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                utils.sample_weights(
                    3*self.dim_model, 4*self.dim_model
                ), name='W_recur'
            )
            self.b_recur = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.scale = theano.shared(
                model_pre_train['scale'], name='scale'
            )
            #
            self.W_mu = theano.shared(
                model_pre_train['W_mu'], name='W_mu'
            )
            #self.delta = theano.shared(
            #    model_pre_train['delta'], name='delta'
            #)
            #
            self.W_delta = theano.shared(
                model_pre_train['W_delta'], name='W_delta'
            )
            #print "W_delta is : "
            #print model_pre_train['W_delta']
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            self.Emb_time = theano.shared(
                model_pre_train['Emb_time'], name='Emb_time'
            )
            #
            self.Threshold_time = theano.shared(
                model_pre_train['Threshold_time'], name='Threshold_time'
            )
            #
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            #self.mu, #self.delta,
            self.scale, # scale parameter
            self.W_mu, self.W_delta, self.W_alpha,
            self.Emb_event, self.Emb_time,
            self.Threshold_time,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        # for intensity eval
        #self.lambda_sum_over_seq = None
        self.lambda_samples = None
        self.num_of_samples = None
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        # tensor[(x == 0).nonzeros()]
        #v_max = numpy.float32(1e9)
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        #a = tensor.switch(z>=v_max, v_max, z)
        #y[(x>=100.0).nonzeros()] = x[(x>=100.0).nonzeros()]
        #np.finfo(np.float32).max
        return z
    #
    #
    def soft_relu_scale(self, x):
        # x is symbolic tensor
        # last dim is dim_process
        # this is important !
        x /= self.scale
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        z *= self.scale
        return z
    #
    #
    def rnn_unit(
        self, emb_event_t, emb_time_t,
        hidden_tm1, cell_tm1
    ):
        pre_transform = tensor.concatenate(
            [emb_event_t, emb_time_t, hidden_tm1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        #
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[:, 3*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_tm1 + gate_input * gate_pre_c
        hidden_t = gate_output * tensor.tanh(cell_t)
        return hidden_t, cell_t
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
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
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        '''
        pass time values through thresholds
        '''
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        # self.W_delta : dim_model * dim_model * dim_process
        #
        delta_for_sims = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_sims, self.W_delta, (2,0)
            )
        )
        #
        # N * size_batch * dim_model
        #
        seq_sims_hidden_with_time = seq_hidden_for_sims[
            :, :, :
        ] * tensor.exp(
            -delta_for_sims * seq_sims_time_to_current[
                :, :, None
            ]
        )
        #
        # N * size_batch * dim_model
        # self.W_alpha : dim_model * dim_process
        mu_for_sims = tensor.tensordot(
            seq_hidden_for_sims, self.W_mu, (2,0)
        )
        # N * size_batch * dim_process
        #
        lambda_over_seq_sims_tilde = mu_for_sims + tensor.tensordot(
            seq_sims_hidden_with_time, self.W_alpha,
            (2, 0)
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu_scale(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        term_2 = numpy.float32(0.0)
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_for_lambda : T * size_batch * dim_model
        #
        #
        delta_for_lambda = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_lambda, self.W_delta, (2,0)
            )
        )
        # T * size_batch * dim_model
        #
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :
        ] * tensor.exp(
            -delta_for_lambda * seq_time_to_current[
                :, :, None
            ]
        )
        # T * size_batch * dim_model
        #
        mu_for_lambda = tensor.tensordot(
            seq_hidden_for_lambda, self.W_mu, (2,0)
        )
        # T * size_batch * dim_process
        #
        lambda_over_seq_tilde = mu_for_lambda + tensor.tensordot(
            seq_hidden_with_time, self.W_alpha,
            (2, 0)
        )
        # T * size_batch * dim_process
        lambda_over_seq = self.soft_relu_scale(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        #
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def compute_lambda(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute intensity
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        '''
        pass time values through thresholds
        '''
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_hidden_for_lambda.shape
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_hidden_for_sims = seq_hidden_for_lambda.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        # seq_sims_time_to_current : N * size_batch
        # self.W_delta : dim_model * dim_model * dim_process
        #
        delta_for_sims = self.soft_relu(
            tensor.tensordot(
                seq_hidden_for_sims, self.W_delta, (2,0)
            )
        )
        #
        # N * size_batch * dim_model
        #
        seq_sims_hidden_with_time = seq_hidden_for_sims[
            :, :, :
        ] * tensor.exp(
            -delta_for_sims * seq_sims_time_to_current[
                :, :, None
            ]
        )
        #
        # N * size_batch * dim_model
        # self.W_alpha : dim_model * dim_process
        mu_for_sims = tensor.tensordot(
            seq_hidden_for_sims, self.W_mu, (2,0)
        )
        # N * size_batch * dim_process
        #
        lambda_over_seq_sims_tilde = mu_for_sims + tensor.tensordot(
            seq_sims_hidden_with_time, self.W_alpha,
            (2, 0)
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu_scale(
            lambda_over_seq_sims_tilde
        )
        #
        '''
        this block is to compute intensity
        '''
        self.lambda_samples = lambda_over_seq_sims.transpose((2,0,1)) * seq_sims_mask[None,:,:]
        self.num_of_samples = tensor.sum(seq_sims_mask)
        #
        #
    #
    #
    def compute_prediction_loss(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_mask,
        time_diffs
    ):
        #
        print "computing predictions loss ... "
        seq_emb_event = self.Emb_event[seq_type_event, :]
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        # seq_hidden : (T+1) * size_batch * dim_model
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # seq_hidden_for_lambda :
        # T * size_batch * dim_model
        delta_for_lambda_pre = tensor.tensordot(
            seq_hidden_for_lambda, self.W_delta, (2,0)
        )
        # T * size_batch * dim_model
        delta_for_lambda = self.soft_relu(
            delta_for_lambda_pre
        )
        # T * size_batch * dim_model
        # time_diffs : M * --> a vector
        seq_hidden_with_time = seq_hidden_for_lambda[
            :, :, :, None
        ] * tensor.exp(
            -delta_for_lambda[
                :, :, :, None
            ] * time_diffs[
                None, None, None, :
            ]
        )
        # T * size_batch * dim_model * M
        mu_for_lambda = tensor.tensordot(
            seq_hidden_for_lambda, self.W_mu, (2,0)
        )
        # T * size_batch * dim_process
        lambda_over_seq_tilde = tensor.sum(
            seq_hidden_with_time[
                :, :, :, None, :
            ] * self.W_alpha[
                None, None, :, :, None
            ], axis = 2
        ) + mu_for_lambda[:, :, :, None]
        # T * size_batch * dim_process * M
        # each time stamp, each seq in batch
        # each process, each simulation for prediction
        lambda_over_seq = self.soft_relu_scale(
            lambda_over_seq_tilde.dimshuffle(3,0,1,2)
        ).dimshuffle(1,2,3,0)
        #
        # T * size_batch * dim_process * M
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis=2
        )
        # T * size_batch * M
        term_1 = time_diffs
        # M *
        #
        #
        cum_num = tensor.arange(
            time_diffs.shape[0]+numpy.int32(1)
        )[1:] * numpy.float32(1.0)
        # M
        term_2 = tensor.exp(
            (
                -1.0 * tensor.extra_ops.cumsum(
                    lambda_sum_over_seq, axis = 2
                ) / cum_num[None, None, :]
            ) * time_diffs[
                None, None, :
            ]
        )
        #
        #term_2 = tensor.exp(
        #    -1.0 * lambda_sum_over_seq * time_diffs[
        #        None, None, :
        #    ]
        #)
        # T * size_batch * M
        term_3 = lambda_sum_over_seq
        # T * size_batch * M
        density = term_2 * term_3
        # T * size_batch * M
        time_prediction = tensor.mean(
            term_1[None, None, :] * density,
            axis = 2
        ) * time_diffs[-1]
        # T * size_batch
        lambda_over_seq_over_sims = lambda_over_seq[
            :, :, :, :
        ] * density[
            :, :, None, :
        ] / lambda_sum_over_seq[
            :, :, None, :
        ]
        # T * size_batch * dim_process * M
        prob_over_seq_over_type = tensor.mean(
            lambda_over_seq_over_sims, axis = 3
        ) * time_diffs[-1]
        # T * size_batch * dim_process
        prob_over_seq_over_type /= tensor.sum(
            prob_over_seq_over_type,
            axis=2,
            keepdims=True
        )
        # T * size_batch * dim_process
        #type_prediction = tensor.argmax(
        #    prob_over_seq_over_type, axis = 2
        #)
        # T * size_batch
        # Now we have :
        # time_prediction, type_prediction, seq_mask
        # all of -- T * size_batch
        target_type = seq_type_event[1:, :]
        target_time = seq_time_values[1:, :]
        # Type first
        new_shape_0 = target_type.shape[0] * target_type.shape[1]
        new_shape_1 = self.dim_process
        back_shape_0 = target_type.shape[0]
        back_shape_1 = target_type.shape[1]
        #
        prob_over_seq = prob_over_seq_over_type.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            target_type.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        log_prob_over_seq = tensor.log(
            prob_over_seq + numpy.float32(1e-9)
        )
        log_prob_over_seq *= seq_mask
        self.log_likelihood_type_predict = tensor.sum(
            log_prob_over_seq
        )
        #diff_type = tensor.abs_(
        #    target_type - type_prediction
        #) * seq_mask
        #diff_type = tensor.switch(
        #    diff_type >= numpy.float32(0.5),
        #    numpy.float32(1.0), numpy.float32(0.0)
        #)
        #
        #self.num_of_errors = tensor.sum(diff_type)
        # Time
        diff_time = (
            target_time - time_prediction
        )**2
        diff_time *= seq_mask
        self.square_errors = tensor.sum(diff_time)
        self.num_of_events = tensor.sum(seq_mask)
        #TODO: Hamming loss for prediction checking
        #
        type_prediction = tensor.argmax(
            prob_over_seq_over_type, axis = 2
        )
        diff_type = tensor.abs_(
            target_type - type_prediction
        ) * seq_mask
        diff_type = tensor.switch(
            diff_type >= numpy.float32(0.5),
            numpy.float32(1.0), numpy.float32(0.0)
        )
        self.num_of_errors = tensor.sum(diff_type)
        #
        self.cost_to_optimize = -self.log_likelihood_type_predict / self.num_of_events + self.square_errors / self.num_of_events + self.term_reg
        #self.cost_to_optimize = -self.log_likelihood_type_predict + self.term_reg
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        self.abs_grad_params = 0.0
        for grad_param in self.grad_params:
            self.abs_grad_params += tensor.sum(
                tensor.abs_(
                    grad_param
                )
            )
        #
        #
    #
    #
    #TODO: memory efficient version of prediction loss
    def predict_each_step(
        self, hidden_for_lambda, time_diffs
    ):
        # hidden_for_lambda : size_batch * dim_model
        # time_diffs : M
        delta_for_lambda = self.soft_relu(
            tensor.tensordot(
                hidden_for_lambda, self.W_delta, (1,0)
            )
        )
        # delta_for_lambda : size_batch * dim_model
        hidden_with_time = hidden_for_lambda[
            :, :, None
        ] * tensor.exp(
            -delta_for_lambda[
                :, :, None
            ] * time_diffs[
                None, None, :
            ]
        )
        # hidden_with_time : size_batch * dim_model * M
        mu_for_lambda = tensor.tensordot(
            hidden_for_lambda, self.W_mu, (1,0)
        )
        # mu_for_lambda : size_batch * dim_process
        lambda_tilde = tensor.sum(
            hidden_with_time[
                :, :, None, :
            ] * self.W_alpha[
                None, :, : , None
            ], axis = 1
        ) + mu_for_lambda[:, :, None]
        # size_batch * dim_process * M
        lambda_each_step = self.soft_relu_scale(
            lambda_tilde.dimshuffle(2,0,1)
        ).dimshuffle(1,2,0)
        # size_batch * dim_process * M
        lambda_sum_each_step = tensor.sum(
            lambda_each_step, axis=1
        )
        # size_batch * M
        #TODO: compute integral
        term_1 = time_diffs
        cum_num = tensor.arange(
            time_diffs.shape[0]+numpy.int32(1)
        )[1:] * numpy.float32(1.0)
        # M
        term_2 = tensor.exp(
            (
                -1.0 * tensor.extra_ops.cumsum(
                    lambda_sum_each_step, axis=1
                ) / cum_num[None, :]
            ) * time_diffs[None, :]
        )
        # size_batch * M
        term_3 = lambda_sum_each_step
        density = term_2 * term_3
        # size_batch * M
        time_prediction_each_step = tensor.mean(
            term_1[None, :] * density, axis=1
        ) * time_diffs[-1]
        # size_batch
        lambda_each_step_over_sims = lambda_each_step[
            :, :, :
        ] * density[
            :, None, :
        ] / lambda_sum_each_step[
            :, None, :
        ]
        # size_batch * dim_process * M
        prob_over_type = tensor.mean(
            lambda_each_step_over_sims, axis=2
        ) * time_diffs[-1]
        # size_batch * dim_process
        prob_over_type /= tensor.sum(
            prob_over_type, axis=1, keepdims=True
        )
        # size_batch * dim_process
        return prob_over_type, time_prediction_each_step

    #
    def compute_prediction_loss_lessmem(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_mask,
        time_diffs
    ):
        #
        print "computing predictions loss ... "
        print "memory efficient version ... "
        seq_emb_event = self.Emb_event[seq_type_event, :]
        seq_time_rep = tensor.nnet.relu(
            seq_time_values[:,:,None] - self.Threshold_time[None,None,:]
        ) # T/T+1 * size_batch * dim_time
        #
        seq_time_rep = tensor.concatenate(
            [seq_time_rep, seq_time_values[:,:,None]],
            axis=2
        )
        #
        seq_emb_time = tensor.tensordot(
            seq_time_rep, self.Emb_time, (2,0)
        )
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden, seq_cell], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(input=seq_emb_event, taps=[0]),
                dict(input=seq_emb_time, taps=[0])
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #TODO: get sequence of hidden units
        # seq_hidden : (T+1) * size_batch * dim_model
        seq_hidden_for_lambda = seq_hidden[:-1, :, :]
        # seq_hidden_for_lambda :
        # T * size_batch * dim_model
        #TODO: predict time and type for each step
        [prob_over_seq_over_type, time_prediction], _ = theano.scan(
            fn = self.predict_each_step,
            sequences = dict(
                input=seq_hidden_for_lambda, taps=[0]
            ),
            outputs_info = [
                None, None
            ],
            non_sequences = time_diffs
        )
        #
        target_type = seq_type_event[1:, :]
        target_time = seq_time_values[1:, :]
        # Type first
        new_shape_0 = target_type.shape[0] * target_type.shape[1]
        new_shape_1 = self.dim_process
        back_shape_0 = target_type.shape[0]
        back_shape_1 = target_type.shape[1]
        #
        prob_over_seq = prob_over_seq_over_type.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            target_type.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        log_prob_over_seq = tensor.log(
            prob_over_seq + numpy.float32(1e-9)
        )
        log_prob_over_seq *= seq_mask
        self.log_likelihood_type_predict = tensor.sum(
            log_prob_over_seq
        )
        #
        # Time
        diff_time = (
            target_time - time_prediction
        )**2
        diff_time *= seq_mask
        self.square_errors = tensor.sum(diff_time)
        self.num_of_events = tensor.sum(seq_mask)
        #TODO: Hamming loss for prediction checking
        #
        type_prediction = tensor.argmax(
            prob_over_seq_over_type, axis = 2
        )
        diff_type = tensor.abs_(
            target_type - type_prediction
        ) * seq_mask
        diff_type = tensor.switch(
            diff_type >= numpy.float32(0.5),
            numpy.float32(1.0), numpy.float32(0.0)
        )
        self.num_of_errors = tensor.sum(diff_type)
        #
        self.cost_to_optimize = -self.log_likelihood_type_predict / self.num_of_events + self.square_errors / self.num_of_events + self.term_reg
        #self.cost_to_optimize = -self.log_likelihood_type_predict + self.term_reg
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        self.abs_grad_params = 0.0
        for grad_param in self.grad_params:
            self.abs_grad_params += tensor.sum(
                tensor.abs_(
                    grad_param
                )
            )
        #
        #
    #
    #
    #
    def get_model(self):
        print "getting model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        return model_dict
    #
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
#
#
#
#
#
