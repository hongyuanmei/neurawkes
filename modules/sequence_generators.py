# -*- coding: utf-8 -*-
"""

Here are the sequence generators
including LSTM generator and Hawkes generator

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
import struct

dtype=theano.config.floatX

class HawkesGen(object):
    '''
    here is the sequence generator using Hawkes process
    '''
    def __init__(self, settings):
        '''
        we follow the definition of multivariate Hawkes process
        mu is the base intensity and
        alpha is the effect matrix and
        delta is the decay matrix
        we randomly sample mu, alpha, delta
        '''
        self.args = settings['args']
        self.sum_for_time = settings['sum_for_time']
        numpy.random.seed(
            settings['seed_random']
        )
        print "initializing ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.mu = numpy.float32(
                numpy.random.uniform(
                    low=0.0, high=1.0,
                    size=(self.dim_process,)
                )
            )
            self.alpha = numpy.float32(
                numpy.random.uniform(
                    low=0.0, high=1.0,
                    size=(self.dim_process, self.dim_process)
                )
            )
            self.delta = numpy.float32(
                numpy.random.uniform(
                    low=10.0, high=20.0,
                    size=(self.dim_process, self.dim_process)
                )
            )
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.mu = model_pre_train['mu']
            self.alpha = model_pre_train['alpha']
            self.delta = model_pre_train['delta']
        #self.intensity = numpy.copy(self.mu)
        self.name = 'HawkesGen'
        #
        self.intensity = numpy.copy(self.mu)
        self.one_seq = []
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        print "done "
        #
    #
    def set_params(self):
        print "set the params for missing data experiments ... "
        self.dim_process = numpy.int32(4)
        self.mu = numpy.float32(
            numpy.ones((self.dim_process, ))
        )
        self.alpha = numpy.float32(
            numpy.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0]
                ]
            )
        )
        self.delta = numpy.float32(
            numpy.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0]
                ]
            )
        )
    #
    def set_args(self, dict_args):
        self.args = dict_args
    #
    #
    def save_model(self, file_save):
        print "saving model of generator ... "
        model_dict = {
            'mu': numpy.copy(self.mu),
            'alpha': numpy.copy(self.alpha),
            'delta': numpy.copy(self.delta),
            'dim_process': self.dim_process,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity = numpy.copy(self.mu)
        self.one_seq = []
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        # initiliaze with mu
        # we do not neet to check
        # if time_current exceeds the sequence
        # since it is automatically garanteed
        self.intensity = numpy.copy(self.mu)
        for event in self.one_seq:
            time_since_start = event['time_since_start']
            #if time_current > time_since_start:
                # if this event is counted as * past event *
            type_event = event['type_event']
            change_time = time_current - time_since_start
            decay_frac = numpy.exp(
                -self.delta[:, type_event] * change_time
            )
            # self.intensity += self.alpha[:, idx_to_occur]
            self.intensity += numpy.copy(
                self.alpha[:, type_event] * decay_frac
            )
        # intensity computation is finished
    #
    #
    #
    # we can try using another method to sample data, which is quicker
    # we can first sample a point with rate \sum \lambda
    # and then sample the type based on the \lambda_k
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        self.compute_intensity_given_past(time_current)
        intensity_hazard = numpy.copy(
            self.intensity[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            time_current += E / intensity_hazard
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
            # this snippet below is for adaptive thining
            # it can speed things up
            # by decreasing upper bound
            # but it is closed when data is randomly generated at the beginning of this project
            intensity_hazard = numpy.copy(
                self.intensity[type_event]
            )
        #
        return time_current
        #
    #
    #
    def sample_time_for_all_type(self):
        # type_event is the type of event for which we want to sample the time
        # it is k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        self.compute_intensity_given_past(time_current)
        intensity_hazard = numpy.sum(self.intensity)
        #
        u = 1.5
        while u >= 1.0:
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            time_current += E / intensity_hazard
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / numpy.sum(self.intensity)
            # this snippet below is for adaptive thining
            # it can speed things up
            # by decreasing upper bound
            # but it is toggled off when data is randomly generated at the beginning of this project
            intensity_hazard = numpy.sum(self.intensity)
        #
        return time_current
        #
    #
    #
    def sample_one_event_sep(self):
        time_of_happen = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        for type_event in range(self.dim_process):
            # sample one event using "thinning algorithm"
            time_of_happen[type_event] = numpy.copy(
                self.sample_time_given_type(
                    type_event
                )
            )
        #
        time_since_start_new = numpy.min(time_of_happen)
        type_event_new = numpy.argmin(time_of_happen)
        return time_since_start_new, type_event_new
    #
    #
    def sample_one_event_tog(self):
        time_since_start_new = self.sample_time_for_all_type()
        self.compute_intensity_given_past(
            time_since_start_new
        )
        prob = self.intensity / numpy.sum(self.intensity)
        type_event_new = numpy.random.choice(
            range(self.dim_process), p = prob
        )
        return time_since_start_new, numpy.int32(type_event_new)
    #
    #
    def sample_one_event(self):
        if self.sum_for_time:
            return self.sample_one_event_tog()
        else:
            return self.sample_one_event_sep()
    #
    #
    def gen_one_seq(self, max_len):
        self.restart_sequence()
        #Liiniger (2009), p. 28, describes a "thinning algorithm":
        #generate one event of each type, take the minimum,
        #and discard the others.
        #Details found in my paper write-up
        #
        #max_len is a pre-sampled value to set the length of seq
        # initialize the seq
        time_since_start = numpy.float32(0.0)
        time_since_start_each_event = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        #
        for idx_event in range(max_len):
            time_since_start_new, type_event_new = self.sample_one_event()
            self.cnt_total_event += 1
            #
            # update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[type_event_new]
            time_since_start_each_event[type_event_new] = time_since_start
            self.one_seq.append(
                {
                    'idx_event': self.cnt_total_event,
                    'type_event': type_event_new,
                    'time_since_start': time_since_start,
                    'time_since_last_event': time_since_last_event,
                    'time_since_last_same_event': time_since_last_same_event
                }
            )
            #
        #
        #
        #
    #
    #
    def gen_seqs(self, settings):
        #
        #print settings
        num_seqs = settings['num_seqs']
        #
        self.list_seqs = []
        cnt_seqs = 0
        #for idx_seq in range(num_seqs):
        while cnt_seqs < num_seqs:
            #
            max_len = numpy.int32(
                round(
                    numpy.random.uniform(
                        low=settings['min_len'],
                        high=settings['max_len']
                    )
                )
            )
            #
            self.gen_one_seq(max_len)
            self.list_seqs.append(self.one_seq)
            cnt_seqs += 1
            if cnt_seqs % 10 == 9:
                print "idx seq of gen : ", (cnt_seqs, self.name)
                print "total number of seqs : ", num_seqs
    #
    #
    def print_some(self):
        print "printing some seqs ... "
        for idx_seq in range(10):
            print "the id of this seq is : ", idx_seq
            seq = self.list_seqs[idx_seq]
            list_events = []
            list_time = []
            list_dtime = []
            list_items = []
            for event_item in seq:
                list_events.append(event_item['type_event'])
                list_time.append(
                    round(event_item['time_since_start'], 4)
                )
                list_dtime.append(
                    round(event_item['time_since_last_event'], 4)
                )
                list_items.append(
                    (
                        event_item['type_event'],
                        round(
                            event_item['time_since_last_event'], 4
                        )
                    )
                )
            print "the events, time and diff time for : ", idx_seq
            print list_events
            print list_time
            print list_dtime
            print "the list of items is : "
            print list_items
    #
    #
    def save_seqs(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_seqs, f)


class HawkesInhibGen(object):
    '''
    here is the sequence generator using Hawkes process with inhibition
    '''
    def __init__(self, settings):
        '''
        we follow the definition of multivariate Hawkes process
        mu is the base intensity and
        alpha is the effect matrix and
        delta is the decay matrix
        we randomly sample mu, alpha, delta
        '''
        print "initializing ... "
        self.args = settings['args']
        self.sum_for_time = settings['sum_for_time']
        numpy.random.seed(
            settings['seed_random']
        )
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.mu = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.dim_process,)
                )
            )
            self.alpha = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.dim_process, self.dim_process)
                )
            )
            self.delta = numpy.float32(
                numpy.random.uniform(
                    low=10.0, high=20.0,
                    size=(self.dim_process, self.dim_process)
                )
            )
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.mu = model_pre_train['mu']
            self.alpha = model_pre_train['alpha']
            self.delta = model_pre_train['delta']
        #self.intensity = numpy.copy(self.mu)
        self.name = 'HawkesInhibGen'
        #
        self.intensity_tilde = numpy.copy(self.mu)
        self.intensity = numpy.log(
            numpy.float32(1.0) + numpy.exp(
                self.intensity_tilde
            )
        )
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        print "done "
        #
        #
    #
    #
    def set_args(self, dict_args):
        self.args = dict_args
    #
    #
    def soft_relu(self, x):
        return numpy.log(numpy.float32(1.0)+numpy.exp(x))
    #
    def hard_relu(self, x):
        return numpy.float32(0.5) * (x + numpy.abs(x) )
        #
    #
    def save_model(self, file_save):
        print "saving model of generator ... "
        model_dict = {
            'mu': numpy.copy(self.mu),
            'alpha': numpy.copy(self.alpha),
            'delta': numpy.copy(self.delta),
            'dim_process': self.dim_process,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = numpy.copy(self.mu)
        self.intensity = self.soft_relu(self.intensity_tilde)
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        # initiliaze with mu
        self.intensity_tilde = numpy.copy(self.mu)
        for event in self.one_seq:
            time_since_start = event['time_since_start']
            #if time_current > time_since_start:
                # if this event is counted as * past event *
            type_event = event['type_event']
            change_time = time_current - time_since_start
            decay_frac = numpy.exp(
                -self.delta[:, type_event] * change_time
            )
            # self.intensity += self.alpha[:, idx_to_occur]
            self.intensity_tilde += numpy.copy(
                self.alpha[:, type_event] * decay_frac
            )
        self.intensity = self.soft_relu(
            self.intensity_tilde
        )
        # intensity computation is finished
    #
    #
    def compute_intensity_upper_bound(self, time_current):
        # compute the upper bound of intensity
        # at the current time
        self.intensity_tilde_ub = numpy.copy(
            self.mu
        )
        # to speed up, this mu is not taken relu
        # but it is still a upper bound
        #self.hard_relu(
        #    self.mu
        #)
        for event in self.one_seq:
            time_since_start = event['time_since_start']
            #if time_current > time_since_start:
            type_event = event['type_event']
            change_time = time_current - time_since_start
            decay_frac = numpy.exp(
                -self.delta[:, type_event] * change_time
            )
            self.intensity_tilde_ub += numpy.copy(
                self.hard_relu(
                    self.alpha[:, type_event]
                ) * decay_frac
            )
        self.intensity_ub = self.soft_relu(
            self.intensity_tilde_ub
        )
    #
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        #
        intensity_hazard = numpy.copy(
            self.intensity_ub[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            time_current += ( E / intensity_hazard )
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
            # for adaptive thinning,
            # decrease the upper bound
            # this is not used at the beginning of the project
            # it is only used for sampling given pre-trained models
            self.compute_intensity_upper_bound(time_current)
            intensity_hazard = numpy.copy(
                self.intensity_ub[type_event]
            )
        #
        return time_current
    #
    #
    def sample_time_for_all_type(self):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        intensity_hazard = numpy.sum(self.intensity_ub)
        #
        u = 1.5
        while u >= 1.0:
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            time_current += ( E / intensity_hazard )
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / numpy.sum(self.intensity)
            # for adaptive thinning,
            # decrease the upper bound
            # this is not used at the beginning of the project
            # it is only used for sampling given pre-trained models
            '''
            self.compute_intensity_upper_bound(time_current)
            intensity_hazard = numpy.sum(self.intensity_ub)
            '''
        return time_current
    #
    #
    def sample_one_event_sep(self):
        time_of_happen = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        for type_event in range(self.dim_process):
            # sample one event using "thinning algorithm"
            time_of_happen[type_event] = numpy.copy(
                self.sample_time_given_type(
                    type_event
                )
            )
        #
        time_since_start_new = numpy.min(time_of_happen)
        type_event_new = numpy.argmin(time_of_happen)
        return time_since_start_new, type_event_new
    #
    #
    def sample_one_event_tog(self):
        time_since_start_new = self.sample_time_for_all_type()
        self.compute_intensity_given_past(
            time_since_start_new
        )
        prob = self.intensity / numpy.sum(self.intensity)
        type_event_new = numpy.random.choice(
            range(self.dim_process), p = prob
        )
        return time_since_start_new, numpy.int32(type_event_new)
    #
    #
    def sample_one_event(self):
        if self.sum_for_time:
            return self.sample_one_event_tog()
        else:
            return self.sample_one_event_sep()
    #
    #
    def gen_one_seq(self, max_len):
        self.restart_sequence()
        '''
        Liiniger (2009), p. 28, describes a "thinning algorithm":
        generate one event of each type, take the minimum,
        and discard the others.
        Details found in my paper write-up
        #
        max_len is a pre-sampled value to set the length of seq
        '''
        # initialize the seq
        time_since_start = numpy.float32(0.0)
        time_since_start_each_event = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        #
        for idx_event in range(max_len):
            time_since_start_new, type_event_new = self.sample_one_event()
            self.cnt_total_event += 1
            #
            # update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[type_event_new]
            time_since_start_each_event[type_event_new] = time_since_start
            self.one_seq.append(
                {
                    'idx_event': self.cnt_total_event,
                    'type_event': type_event_new,
                    'time_since_start': time_since_start,
                    'time_since_last_event': time_since_last_event,
                    'time_since_last_same_event': time_since_last_same_event
                }
            )
            #
        #
    #
    #
    def gen_seqs(self, settings):
        #
        #print settings
        num_seqs = settings['num_seqs']
        #
        self.list_seqs = []
        cnt_seqs = 0
        #for idx_seq in range(num_seqs):
        while cnt_seqs < num_seqs:
            #
            max_len = numpy.int32(
                round(
                    numpy.random.uniform(
                        low=settings['min_len'],
                        high=settings['max_len']
                    )
                )
            )
            #
            self.gen_one_seq(max_len)
            self.list_seqs.append(self.one_seq)
            cnt_seqs += 1
            if cnt_seqs % 10 == 9:
                print "idx seq of gen : ", (cnt_seqs, self.name)
                print "total number of seqs : ", num_seqs
    #
    #
    def print_some(self):
        print "printing some seqs ... "
        for idx_seq in range(10):
            print "the id of this seq is : ", idx_seq
            seq = self.list_seqs[idx_seq]
            list_events = []
            list_time = []
            list_dtime = []
            list_items = []
            for event_item in seq:
                list_events.append(event_item['type_event'])
                list_time.append(
                    round(event_item['time_since_start'], 4)
                )
                list_dtime.append(
                    round(event_item['time_since_last_event'], 4)
                )
                list_items.append(
                    (
                        event_item['type_event'],
                        round(
                            event_item['time_since_last_event'], 4
                        )
                    )
                )
            print "the events, time and diff time for : ", idx_seq
            print list_events
            print list_time
            print list_dtime
            print "the list of items is : "
            print list_items
    #
    #
    def save_seqs(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_seqs, f)


class NeuralHawkesCTLSTM(object):
    '''
    here is the sequence generator
    using Neural Hawkes process with continuous-time LSTM
    '''
    def __init__(self, settings):
        #
        print "initializing generator ... "
        self.args = settings['args']
        self.sum_for_time = settings['sum_for_time']
        self.dim_float = numpy.int32(32)
        if settings['path_pre_train'] == None:
            print "random parameters ... "
            self.dim_process = settings['dim_process']
            self.dim_model = settings['dim_LSTM']
            self.dim_time = self.dim_float
            #
            numpy.random.seed(
                settings['seed_random']
            )
            #
            #self.scale = numpy.float32(
            #    numpy.random.uniform(
            #        low = 1e-3, high = 2.0,
            #        size = (self.dim_process, )
            #    )
            #)
            self.scale = numpy.float32(
                numpy.ones( (self.dim_process, ) )
            )
            #
            self.W_alpha = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.dim_model, self.dim_process)
                )
            )
            self.Emb_event = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        self.dim_process + numpy.int32(1),
                        self.dim_model
                    )
                )
            )
            self.W_recur = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        2 * self.dim_model,
                        7 * self.dim_model
                    )
                )
            )
            self.b_recur = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (7 * self.dim_model, )
                )
            )
        else:
            print "read pretrained model ... "
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.scale = model_pre_train['scale']
            self.W_alpha = model_pre_train['W_alpha']
            self.Emb_event = model_pre_train['Emb_event']
            self.W_recur = model_pre_train['W_recur']
            self.b_recur = model_pre_train['b_recur']
            #
        #
        #self.intensity = numpy.copy(self.mu)
        self.name = 'NeuralHawkesGenCTLSTM'
        #
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        #self.hidden_t = numpy.zeros(
        #    (self.dim_model, ), dtype = dtype
        #)
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_target = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_decay = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.gate_output = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq) )
        print "initialization done "
        #
    #
    def set_args(self, dict_args):
        self.args = dict_args
    #
    def soft_relu(self, x):
        return numpy.log(numpy.float32(1.0)+numpy.exp(x))
    #
    def soft_relu_scale(self, x):
        # last dim of x is dim_process
        x /= self.scale
        y = numpy.log(numpy.float32(1.0)+numpy.exp(x))
        y *= self.scale
        return y
    #
    def hard_relu(self, x):
        return numpy.float32(0.5) * (x + numpy.abs(x) )
        #
    #
    def save_model(self, file_save):
        print "saving model of generator ... "
        model_dict = {
            'scale': numpy.copy(self.scale),
            'W_alpha': numpy.copy(self.W_alpha),
            'Emb_event': numpy.copy(self.Emb_event),
            'W_recur': numpy.copy(self.W_recur),
            'b_recur': numpy.copy(self.b_recur),
            'dim_process': self.dim_process,
            'dim_model': self.dim_model,
            'dim_time': self.dim_time,
            'dim_float': self.dim_float,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        #self.hidden_t = numpy.zeros(
        #    (self.dim_model, ), dtype = dtype
        #)
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_target = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_decay = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.gate_output = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq) )
        #
    #
    #
    def sigmoid(self, x):
        return 1 / (1+numpy.exp(-x))
    #
    #
    def compute_hidden_states(self):
        # every time it is called,
        # it computes the new hidden states of the LSTM
        # it gets the last event in the sequence
        # which is generated at t_(rec(t))
        # and compute its hidden states
        # Note : for this event, we get its type
        # and time elapsed since last event
        # that is to say, this func is different than
        # rnn_unit in models
        # THERE : event, time_since_this_event_to_next
        # so first update, and then decay
        # HERE : time_since_last_event, event
        # so first decay, and then update
        # Note : this should be called
        # after one event is generated and appended
        # so the state is updated accordingly
        #TODO: decay
        cell_t_after_decay = self.cell_target + (
            self.cell_t - self.cell_target
        ) * numpy.exp(
            -self.cell_decay * self.one_seq[-1][
                'time_since_last_event'
            ]
        )
        hidden_t_after_decay = self.gate_output * numpy.tanh(
            cell_t_after_decay
        )
        #TODO: update
        emb_event_t = self.Emb_event[
            self.one_seq[-1]['type_event'], :
        ]
        post_transform = numpy.dot(
            numpy.concatenate(
                (emb_event_t, hidden_t_after_decay),
                axis = 0
            ), self.W_recur
        ) + self.b_recur
        #
        gate_input = self.sigmoid(
            post_transform[:self.dim_model]
        )
        gate_forget = self.sigmoid(
            post_transform[self.dim_model:2*self.dim_model]
        )
        gate_output = self.sigmoid(
            post_transform[2*self.dim_model:3*self.dim_model]
        )
        gate_pre_c = numpy.tanh(
            post_transform[3*self.dim_model:4*self.dim_model]
        )
        # 2 -- input_bar and forget_bar gates
        gate_input_target = self.sigmoid(
            post_transform[4*self.dim_model:5*self.dim_model]
        )
        gate_forget_target = self.sigmoid(
            post_transform[5*self.dim_model:6*self.dim_model]
        )
        # cell memory decay
        cell_decay = self.soft_relu(
            post_transform[6*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_t_after_decay + gate_input * gate_pre_c
        cell_target = gate_forget_target * self.cell_target + gate_input_target * gate_pre_c
        #
        self.cell_t = numpy.copy(cell_t)
        self.cell_target = numpy.copy(cell_target)
        self.cell_decay = numpy.copy(cell_decay)
        self.gate_output = numpy.copy(gate_output)
        #
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        time_recent = self.one_seq[-1]['time_since_start']
        #
        cell_t_after_decay = self.cell_target + (
            self.cell_t - self.cell_target
        ) * numpy.exp(
            -self.cell_decay * (
                time_current - time_recent
            )
        )
        hidden_t_after_decay = self.gate_output * numpy.tanh(
            cell_t_after_decay
        )
        #
        self.intensity_tilde = numpy.dot(
            hidden_t_after_decay, self.W_alpha
        )
        self.intensity = self.soft_relu_scale(
            self.intensity_tilde
        )
        # intensity computation is finished
        #
    #
    #
    def compute_intensity_upper_bound(self, time_current):
        # compute the upper bound of intensity
        # at the current time
        # Note : this is very tricky !!!
        # in decomposable process, finding upper bound is easy
        # see B.3 in NIPS paper
        # but in neural model
        # it is not a combo of POSITIVE decreasing funcs
        # So how to do this?
        # we find the functon is a sum of temrs
        # some terms are decreasing, we keep them
        # some terms are increasing, we get their upper-limit
        #
        # In detail, we compose it to 4 parts :
        # (dc = c-c_target)
        # w + dc -  increasing
        # w + dc +  decreasing
        # w - dc -  decreasing
        # w - dc +  increasing
        #
        time_recent = self.one_seq[-1]['time_since_start']
        #
        cell_gap = self.cell_t - self.cell_target
        cell_gap_matrix = numpy.outer(
            cell_gap, numpy.ones(
                (self.dim_process, ), dtype=dtype
            )
        )
        # dim * dim_process
        index_increasing_0 = (cell_gap_matrix > 0.0) & (self.W_alpha < 0.0)
        index_increasing_1 = (cell_gap_matrix < 0.0) & (self.W_alpha > 0.0)
        #
        cell_gap_matrix[
            index_increasing_0
        ] = numpy.float32(0.0)
        cell_gap_matrix[
            index_increasing_1
        ] = numpy.float32(0.0)
        #
        cell_t_after_decay = numpy.outer(
            self.cell_target, numpy.ones(
                (self.dim_process, ), dtype=dtype
            )
        ) + cell_gap_matrix * numpy.exp(
            -numpy.outer(
                self.cell_decay, numpy.ones(
                    (self.dim_process, ), dtype=dtype
                )
            ) * (
                time_current - time_recent
            )
        )
        hidden_t_after_decay = numpy.outer(
            self.gate_output, numpy.ones(
                (self.dim_process, ), dtype=dtype
            )
        ) * numpy.tanh(cell_t_after_decay)
        #
        self.intensity_tilde_ub = numpy.sum(
            hidden_t_after_decay * self.W_alpha,
            axis=0
        )
        self.intensity_ub = self.soft_relu_scale(
            self.intensity_tilde_ub
        )
        #
        # intensity computation is finished
    #
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        intensity_hazard = numpy.copy(
            self.intensity_ub[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            #print "type is : ", type_event
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            #print "E U time_current : "
            #print E, U, time_current
            #print "intensity hazard is : "
            #print intensity_hazard
            time_current += (E / intensity_hazard)
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
            #print "new time_current and u : "
            #print time_current, u
            #print "intensity and upper bound is : "
            #print self.intensity
            #print self.intensity_ub
            # use adaptive thinning algorithm
            # that is, decreasing the upper bound
            # to make the sampling quicker
            # use adaptive method by
            # toggling on the following block
            '''
            self.compute_intensity_upper_bound(
                time_current
            )
            intensity_hazard = numpy.copy(
                self.intensity_ub[type_event]
            )
            '''
        return time_current
        #
    #
    #
    def sample_time_for_all_type(self):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        intensity_hazard = numpy.sum(self.intensity_ub)
        #
        u = 1.5
        while u >= 1.0:
            #print "type is : ", type_event
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            #print "E U time_current : "
            #print E, U, time_current
            #print "intensity hazard is : "
            #print intensity_hazard
            time_current += (E / intensity_hazard)
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / numpy.sum(self.intensity)
            #print "new time_current and u : "
            #print time_current, u
            #print "intensity and upper bound is : "
            #print self.intensity
            #print self.intensity_ub
            # use adaptive thinning algorithm
            # that is, decreasing the upper bound
            # to make the sampling quicker
            # use adaptive method by
            # toggling on the following block
            '''
            self.compute_intensity_upper_bound(
                time_current
            )
            intensity_hazard = numpy.sum(self.intensity_ub)
            '''
        return time_current
        #
    #
    #
    def sample_one_event_sep(self):
        time_of_happen = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        for type_event in range(self.dim_process):
            # sample one event using "thinning algorithm"
            time_of_happen[type_event] = numpy.copy(
                self.sample_time_given_type(
                    type_event
                )
            )
        #
        time_since_start_new = numpy.min(time_of_happen)
        type_event_new = numpy.argmin(time_of_happen)
        return time_since_start_new, type_event_new
    #
    #
    def sample_one_event_tog(self):
        time_since_start_new = self.sample_time_for_all_type()
        self.compute_intensity_given_past(
            time_since_start_new
        )
        prob = self.intensity / numpy.sum(self.intensity)
        type_event_new = numpy.random.choice(
            range(self.dim_process), p = prob
        )
        return time_since_start_new, numpy.int32(type_event_new)
    #
    #
    def sample_one_event(self):
        if self.sum_for_time:
            return self.sample_one_event_tog()
        else:
            return self.sample_one_event_sep()
    #
    #
    def gen_one_seq(self, max_len):
        self.restart_sequence()
        '''
        Liiniger (2009), p. 28, describes a "thinning algorithm":
        generate one event of each type, take the minimum,
        and discard the others.
        Details found in NIPS 17 Appendix
        max_len is a pre-sampled value to set the length of seq
        '''
        # initialize the seq
        time_since_start = numpy.float32(0.0)
        time_since_start_each_event = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        #
        for idx_event in range(max_len):
            #
            # compute the hidden states
            # of the most recent event in sequence
            self.compute_hidden_states()
            #
            time_since_start_new, type_event_new = self.sample_one_event()
            self.cnt_total_event += 1
            #
            # update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[type_event_new]
            time_since_start_each_event[type_event_new] = time_since_start
            self.one_seq.append(
                {
                    'idx_event': self.cnt_total_event,
                    'type_event': type_event_new,
                    'time_since_start': time_since_start,
                    'time_since_last_event': time_since_last_event,
                    'time_since_last_same_event': time_since_last_same_event
                }
            )
            #
        # throw away the BOS item
        # at the head of the sequence
        self.one_seq.pop(0)
        #
    #
    #
    def gen_seqs(self, settings):
        #print settings
        print "generating sequences ... "
        num_seqs = settings['num_seqs']
        #
        self.list_seqs = []
        cnt_seqs = 0
        #for idx_seq in range(num_seqs):
        while cnt_seqs < num_seqs:
            #
            max_len = numpy.int32(
                round(
                    numpy.random.uniform(
                        low=settings['min_len'],
                        high=settings['max_len']
                    )
                )
            )
            #
            self.gen_one_seq(max_len)
            self.list_seqs.append(self.one_seq)
            cnt_seqs += 1
            if cnt_seqs % 10 == 9:
                print "idx seq of gen : ", (cnt_seqs, self.name)
                print "total number of seqs : ", num_seqs
    #
    #
    def print_some(self):
        print "printing some seqs ... "
        for idx_seq in range(10):
            print "the id of this seq is : ", idx_seq
            seq = self.list_seqs[idx_seq]
            list_events = []
            list_time = []
            list_dtime = []
            list_items = []
            for event_item in seq:
                list_events.append(event_item['type_event'])
                list_time.append(
                    round(event_item['time_since_start'], 4)
                )
                list_dtime.append(
                    round(event_item['time_since_last_event'], 4)
                )
                list_items.append(
                    (
                        event_item['type_event'],
                        round(
                            event_item['time_since_last_event'], 4
                        )
                    )
                )
            print "the events, time and diff time for : ", idx_seq
            print list_events
            print list_time
            print list_dtime
            print "the list of items is : "
            print list_items
    #
    #
    def save_seqs(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_seqs, f)
#
#
#
#
#
#
# deprecated generators
# TODO: modules below are deprecated
# they are models that we tried over this project
# most of them work, better than Hawkes baseline
# but still lose to our neural Hawkes with continuous-time LSTM
# most of them keep the decomposable structure of Hawkes
# and try to use neural networks to parametrize it
#
#

class NeuralHawkesGen(object):
    '''
    here is the sequence generator using Neural Hawkes process
    '''
    def __init__(self, settings):
        #
        self.dim_process = settings['dim_process']
        self.dim_model = settings['dim_LSTM']
        #
        self.dim_float = numpy.int32(32)
        self.dim_time = self.dim_float
        #
        self.args = settings['args']
        numpy.random.seed(
            settings['seed_random']
        )
        self.mu = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (self.dim_process,)
            )
        )
        #
        self.delta = numpy.float32(
            numpy.random.uniform(
                low=10.0, high=20.0,
                size=(self.dim_model, self.dim_process)
            )
        )
        #
        self.W_alpha = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (self.dim_model, self.dim_process)
            )
        )
        self.Emb_event = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (
                    self.dim_process + numpy.int32(1),
                    self.dim_model
                )
            )
        )
        self.Emb_time = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (
                    self.dim_time, self.dim_model
                )
            )
        )
        self.W_recur = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (
                    3 * self.dim_model,
                    4 * self.dim_model
                )
            )
        )
        self.b_recur = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (4*self.dim_model, )
            )
        )
        #
        #self.intensity = numpy.copy(self.mu)
        self.name = 'NeuralHawkesGen'
        #
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    def soft_relu(self, x):
        return numpy.log(numpy.float32(1.0)+numpy.exp(x))
    #
    def hard_relu(self, x):
        return numpy.float32(0.5) * (x + numpy.abs(x) )
        #
    #
    def save_model(self, file_save):
        print "saving model of generator ... "
        model_dict = {
            'mu': numpy.copy(self.mu),
            'delta': numpy.copy(self.delta),
            'W_alpha': numpy.copy(self.W_alpha),
            'Emb_event': numpy.copy(self.Emb_event),
            'Emb_time': numpy.copy(self.Emb_time),
            'W_recur': numpy.copy(self.W_recur),
            'b_recur': numpy.copy(self.b_recur),
            'dim_process': self.dim_process,
            'dim_model': self.dim_model,
            'dim_time': self.dim_time,
            'dim_float': self.dim_float,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        #
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    #
    #
    def float32_to_bit(self, float_input):
        '''
        input a number in float, convert it to float32
        get its 32-bit representations
        '''
        float32_input = numpy.float32(float_input)
        str_input = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', float32_input))
        bit_input = numpy.zeros(
            (self.dim_float,), dtype=dtype
        )
        assert(self.dim_float == len(str_input))
        for idx, item_in_input in enumerate(str_input):
            bit_input[idx] = numpy.float32(item_in_input)
        return numpy.copy(bit_input)
    #
    #
    def sigmoid(self, x):
        return 1 / (1+numpy.exp(-x))
    #
    #
    def compute_hidden_states(self):
        # every time it is called,
        # it computes the new hidden states of the LSTM
        # it gets the last event in the sequence
        # which is generated at t_(rec(t))
        # and compute its hidden states
        emb_event_t = self.Emb_event[
            self.one_seq[-1]['type_event'], :
        ]
        emb_time_t = numpy.dot(
            self.float32_to_bit(
                self.one_seq[-1]['time_since_last_event']
            ),
            self.Emb_time
        )
        post_transform = numpy.dot(
            numpy.concatenate(
                (emb_event_t, emb_time_t, self.hidden_t),
                axis = 0
            ),
            self.W_recur
        ) + self.b_recur
        #
        gate_input = self.sigmoid(
            post_transform[:self.dim_model]
        )
        gate_forget = self.sigmoid(
            post_transform[self.dim_model:2*self.dim_model]
        )
        gate_output = self.sigmoid(
            post_transform[2*self.dim_model:3*self.dim_model]
        )
        gate_pre_c = numpy.tanh(
            post_transform[3*self.dim_model:]
        )
        #
        cell_t_new = gate_forget * self.cell_t + gate_input * gate_pre_c
        hidden_t_new = gate_output * numpy.tanh(cell_t_new)
        self.hidden_t = numpy.copy(hidden_t_new)
        self.cell_t = numpy.copy(cell_t_new)
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        #
        time_recent = self.one_seq[-1]['time_since_start']
        #
        hidden_with_time = numpy.exp(
            -self.delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde = numpy.sum(
            self.W_alpha * hidden_with_time,
            axis = 0
        ) + self.mu
        #
        self.intensity = self.soft_relu(
            self.intensity_tilde
        )
        # intensity computation is finished
    #
    #
    def compute_intensity_upper_bound(self, time_current):
        # compute the upper bound of intensity
        # at the current time
        time_recent = self.one_seq[-1]['time_since_start']
        #
        hidden_with_time = numpy.exp(
            -self.delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde_ub = numpy.sum(
            self.hard_relu(
                self.W_alpha * hidden_with_time
            ),
            axis = 0
        ) + self.hard_relu(self.mu)
        #
        self.intensity_ub = self.soft_relu(
            self.intensity_tilde_ub
        )
        # intensity computation is finished
    #
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        #
        intensity_hazard = numpy.copy(
            self.intensity_ub[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            time_current += E / intensity_hazard
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
        #
        return time_current
        #
    #
    #
    def gen_one_seq(self, max_len):
        self.restart_sequence()
        '''
        Liiniger (2009), p. 28, describes a "thinning algorithm":
        generate one event of each type, take the minimum,
        and discard the others.
        Details found in my paper write-up
        #
        max_len is a pre-sampled value to set the length of seq
        '''
        # initialize the seq
        time_since_start = numpy.float32(0.0)
        time_since_start_each_event = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        #
        for idx_event in range(max_len):
            time_of_happen = numpy.zeros(
                (self.dim_process,), dtype=dtype
            )
            #
            # compute the hidden states
            # of the most recent event in sequence
            self.compute_hidden_states()
            #
            for type_event in range(self.dim_process):
                # sample one event using "thinning algorithm"
                time_of_happen[type_event] = numpy.copy(
                    self.sample_time_given_type(
                        type_event
                    )
                )
            #
            time_since_start_new = numpy.min(time_of_happen)
            type_event_new = numpy.argmin(time_of_happen)
            self.cnt_total_event += 1
            #
            # update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[type_event_new]
            time_since_start_each_event[type_event_new] = time_since_start
            self.one_seq.append(
                {
                    'idx_event': self.cnt_total_event,
                    'type_event': type_event_new,
                    'time_since_start': time_since_start,
                    'time_since_last_event': time_since_last_event,
                    'time_since_last_same_event': time_since_last_same_event
                }
            )
            #
        # throw away the BOS item
        # at the head of the sequence
        self.one_seq.pop(0)
        #
    #
    #
    def gen_seqs(self, settings):
        #
        #print settings
        num_seqs = settings['num_seqs']
        #
        self.list_seqs = []
        cnt_seqs = 0
        #for idx_seq in range(num_seqs):
        while cnt_seqs < num_seqs:
            #
            max_len = numpy.int32(
                round(
                    numpy.random.uniform(
                        low=settings['min_len'],
                        high=settings['max_len']
                    )
                )
            )
            #
            self.gen_one_seq(max_len)
            self.list_seqs.append(self.one_seq)
            cnt_seqs += 1
            if cnt_seqs % 10 == 9:
                print "idx seq of gen : ", (cnt_seqs, self.name)
                print "total number of seqs : ", num_seqs
    #
    #
    def print_some(self):
        print "printing some seqs ... "
        for idx_seq in range(10):
            print "the id of this seq is : ", idx_seq
            seq = self.list_seqs[idx_seq]
            list_events, list_time = [], []
            for event_item in seq:
                list_events.append(event_item['type_event'])
                list_time.append(
                    round(event_item['time_since_start'], 4)
                )
            print list_events
            print list_time
    #
    def save_seqs(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_seqs, f)


class GeneralizedNeuralHawkesGen(object):
    '''
    here is the sequence generator using Neural Hawkes process
    '''
    def __init__(self, settings):
        #
        self.dim_process = settings['dim_process']
        self.dim_model = settings['dim_LSTM']
        #
        self.dim_float = numpy.int32(32)
        self.dim_time = self.dim_float
        #
        self.args = settings['args']
        numpy.random.seed(
            settings['seed_random']
        )
        self.mu = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (self.dim_process,)
            )
        )
        #
        self.W_delta = numpy.float32(
            numpy.random.uniform(
                low = -1.0, high= 1.0,
                size=(
                    self.dim_model, self.dim_model,
                    self.dim_process
                )
            )
        )
        #
        self.W_alpha = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (self.dim_model, self.dim_process)
            )
        )
        self.Emb_event = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (
                    self.dim_process + numpy.int32(1),
                    self.dim_model
                )
            )
        )
        self.Emb_time = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (
                    self.dim_time, self.dim_model
                )
            )
        )
        self.W_recur = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (
                    3 * self.dim_model,
                    4 * self.dim_model
                )
            )
        )
        self.b_recur = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (4*self.dim_model, )
            )
        )
        #
        #self.intensity = numpy.copy(self.mu)
        self.name = 'GeneralizedNeuralHawkesGen'
        #
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    def soft_relu(self, x):
        return numpy.log(numpy.float32(1.0)+numpy.exp(x))
    #
    def hard_relu(self, x):
        return numpy.float32(0.5) * (x + numpy.abs(x) )
        #
    #
    def save_model(self, file_save):
        print "saving model of generator ... "
        model_dict = {
            'mu': numpy.copy(self.mu),
            'W_delta': numpy.copy(self.W_delta),
            'W_alpha': numpy.copy(self.W_alpha),
            'Emb_event': numpy.copy(self.Emb_event),
            'Emb_time': numpy.copy(self.Emb_time),
            'W_recur': numpy.copy(self.W_recur),
            'b_recur': numpy.copy(self.b_recur),
            'dim_process': self.dim_process,
            'dim_model': self.dim_model,
            'dim_time': self.dim_time,
            'dim_float': self.dim_float,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        #
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    #
    #
    def float32_to_bit(self, float_input):
        '''
        input a number in float, convert it to float32
        get its 32-bit representations
        '''
        float32_input = numpy.float32(float_input)
        str_input = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', float32_input))
        bit_input = numpy.zeros(
            (self.dim_float,), dtype=dtype
        )
        assert(self.dim_float == len(str_input))
        for idx, item_in_input in enumerate(str_input):
            bit_input[idx] = numpy.float32(item_in_input)
        return numpy.copy(bit_input)
    #
    #
    def sigmoid(self, x):
        return 1 / (1+numpy.exp(-x))
    #
    #
    def compute_hidden_states(self):
        # every time it is called,
        # it computes the new hidden states of the LSTM
        # it gets the last event in the sequence
        # which is generated at t_(rec(t))
        # and compute its hidden states
        emb_event_t = self.Emb_event[
            self.one_seq[-1]['type_event'], :
        ]
        emb_time_t = numpy.dot(
            self.float32_to_bit(
                self.one_seq[-1]['time_since_last_event']
            ),
            self.Emb_time
        )
        post_transform = numpy.dot(
            numpy.concatenate(
                (emb_event_t, emb_time_t, self.hidden_t),
                axis = 0
            ),
            self.W_recur
        ) + self.b_recur
        #
        gate_input = self.sigmoid(
            post_transform[:self.dim_model]
        )
        gate_forget = self.sigmoid(
            post_transform[self.dim_model:2*self.dim_model]
        )
        gate_output = self.sigmoid(
            post_transform[2*self.dim_model:3*self.dim_model]
        )
        gate_pre_c = numpy.tanh(
            post_transform[3*self.dim_model:]
        )
        #
        cell_t_new = gate_forget * self.cell_t + gate_input * gate_pre_c
        hidden_t_new = gate_output * numpy.tanh(cell_t_new)
        self.hidden_t = numpy.copy(hidden_t_new)
        self.cell_t = numpy.copy(cell_t_new)
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        #
        time_recent = self.one_seq[-1]['time_since_start']
        #
        delta = self.soft_relu(
            numpy.tensordot(
                self.hidden_t, self.W_delta, (0, 0)
            )
        )
        #
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde = numpy.sum(
            self.W_alpha * hidden_with_time,
            axis = 0
        ) + self.mu
        #
        self.intensity = self.soft_relu(
            self.intensity_tilde
        )
        # intensity computation is finished
    #
    def compute_intensity_upper_bound(self, time_current):
        # compute the upper bound of intensity
        # at the current time
        time_recent = self.one_seq[-1]['time_since_start']
        #
        delta = self.soft_relu(
            numpy.tensordot(
                self.hidden_t, self.W_delta, (0, 0)
            )
        )
        #
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde_ub = numpy.sum(
            self.hard_relu(
                self.W_alpha * hidden_with_time
            ),
            axis = 0
        ) + self.hard_relu(self.mu)
        #
        self.intensity_ub = self.soft_relu(
            self.intensity_tilde_ub
        )
        # intensity computation is finished
    #
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        #
        intensity_hazard = numpy.copy(
            self.intensity_ub[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            time_current += E / intensity_hazard
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
        #
        return time_current
        #
    #
    #
    def gen_one_seq(self, max_len):
        self.restart_sequence()
        '''
        Liiniger (2009), p. 28, describes a "thinning algorithm":
        generate one event of each type, take the minimum,
        and discard the others.
        Details found in my paper write-up
        #
        max_len is a pre-sampled value to set the length of seq
        '''
        # initialize the seq
        time_since_start = numpy.float32(0.0)
        time_since_start_each_event = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        #
        for idx_event in range(max_len):
            time_of_happen = numpy.zeros(
                (self.dim_process,), dtype=dtype
            )
            #
            # compute the hidden states
            # of the most recent event in sequence
            self.compute_hidden_states()
            #
            for type_event in range(self.dim_process):
                # sample one event using "thinning algorithm"
                time_of_happen[type_event] = numpy.copy(
                    self.sample_time_given_type(
                        type_event
                    )
                )
            #
            time_since_start_new = numpy.min(time_of_happen)
            type_event_new = numpy.argmin(time_of_happen)
            self.cnt_total_event += 1
            #
            # update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[type_event_new]
            time_since_start_each_event[type_event_new] = time_since_start
            self.one_seq.append(
                {
                    'idx_event': self.cnt_total_event,
                    'type_event': type_event_new,
                    'time_since_start': time_since_start,
                    'time_since_last_event': time_since_last_event,
                    'time_since_last_same_event': time_since_last_same_event
                }
            )
            #
        # throw away the BOS item
        # at the head of the sequence
        self.one_seq.pop(0)
        #
    #
    #
    def gen_seqs(self, settings):
        #
        #print settings
        num_seqs = settings['num_seqs']
        #
        self.list_seqs = []
        cnt_seqs = 0
        #for idx_seq in range(num_seqs):
        while cnt_seqs < num_seqs:
            #
            max_len = numpy.int32(
                round(
                    numpy.random.uniform(
                        low=settings['min_len'],
                        high=settings['max_len']
                    )
                )
            )
            #
            self.gen_one_seq(max_len)
            self.list_seqs.append(self.one_seq)
            cnt_seqs += 1
            if cnt_seqs % 10 == 9:
                print "idx seq of gen : ", (cnt_seqs, self.name)
                print "total number of seqs : ", num_seqs
    #
    #
    def print_some(self):
        print "printing some seqs ... "
        for idx_seq in range(10):
            print "the id of this seq is : ", idx_seq
            seq = self.list_seqs[idx_seq]
            list_events, list_time = [], []
            for event_item in seq:
                list_events.append(event_item['type_event'])
                list_time.append(
                    round(event_item['time_since_start'], 4)
                )
            print list_events
            print list_time
    #
    def save_seqs(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_seqs, f)


class NeuralHawkesAdaptiveBaseGen(object):
    '''
    here is the sequence generator using Neural Hawkes process
    '''
    def __init__(self, settings):
        #
        self.dim_process = settings['dim_process']
        self.dim_model = settings['dim_LSTM']
        #
        self.dim_float = numpy.int32(32)
        self.dim_time = self.dim_float
        #
        self.args = settings['args']
        numpy.random.seed(
            settings['seed_random']
        )
        self.W_mu = numpy.float32(
            numpy.random.uniform(
                low = -1.0, high = 1.0,
                size = (
                    self.dim_model, self.dim_process
                )
            )
        )
        #
        self.W_delta = numpy.float32(
            numpy.random.uniform(
                low = -1.0, high = 1.0,
                size=(
                    self.dim_model, self.dim_model,
                    self.dim_process
                )
            )
        )
        #
        self.W_alpha = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (self.dim_model, self.dim_process)
            )
        )
        self.Emb_event = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (
                    self.dim_process + numpy.int32(1),
                    self.dim_model
                )
            )
        )
        self.Emb_time = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (
                    self.dim_time, self.dim_model
                )
            )
        )
        self.W_recur = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (
                    3 * self.dim_model,
                    4 * self.dim_model
                )
            )
        )
        self.b_recur = numpy.float32(
            numpy.random.uniform(
                low = -1.0,
                high = 1.0,
                size = (4*self.dim_model, )
            )
        )
        #
        #self.intensity = numpy.copy(self.mu)
        self.name = 'AdaptiveNeuralHawkesGen'
        #
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    def soft_relu(self, x):
        return numpy.log(numpy.float32(1.0)+numpy.exp(x))
    #
    def hard_relu(self, x):
        return numpy.float32(0.5) * (x + numpy.abs(x) )
        #
    #
    def save_model(self, file_save):
        print "saving model of generator ... "
        model_dict = {
            'W_mu': numpy.copy(self.W_mu),
            'W_delta': numpy.copy(self.W_delta),
            'W_alpha': numpy.copy(self.W_alpha),
            'Emb_event': numpy.copy(self.Emb_event),
            'Emb_time': numpy.copy(self.Emb_time),
            'W_recur': numpy.copy(self.W_recur),
            'b_recur': numpy.copy(self.b_recur),
            'dim_process': self.dim_process,
            'dim_model': self.dim_model,
            'dim_time': self.dim_time,
            'dim_float': self.dim_float,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        #
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    #
    #
    def float32_to_bit(self, float_input):
        '''
        input a number in float, convert it to float32
        get its 32-bit representations
        '''
        float32_input = numpy.float32(float_input)
        str_input = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', float32_input))
        bit_input = numpy.zeros(
            (self.dim_float,), dtype=dtype
        )
        assert(self.dim_float == len(str_input))
        for idx, item_in_input in enumerate(str_input):
            bit_input[idx] = numpy.float32(item_in_input)
        return numpy.copy(bit_input)
    #
    #
    def sigmoid(self, x):
        return 1 / (1+numpy.exp(-x))
    #
    #
    def compute_hidden_states(self):
        # every time it is called,
        # it computes the new hidden states of the LSTM
        # it gets the last event in the sequence
        # which is generated at t_(rec(t))
        # and compute its hidden states
        emb_event_t = self.Emb_event[
            self.one_seq[-1]['type_event'], :
        ]
        emb_time_t = numpy.dot(
            self.float32_to_bit(
                self.one_seq[-1]['time_since_last_event']
            ),
            self.Emb_time
        )
        post_transform = numpy.dot(
            numpy.concatenate(
                (emb_event_t, emb_time_t, self.hidden_t),
                axis = 0
            ),
            self.W_recur
        ) + self.b_recur
        #
        gate_input = self.sigmoid(
            post_transform[:self.dim_model]
        )
        gate_forget = self.sigmoid(
            post_transform[self.dim_model:2*self.dim_model]
        )
        gate_output = self.sigmoid(
            post_transform[2*self.dim_model:3*self.dim_model]
        )
        gate_pre_c = numpy.tanh(
            post_transform[3*self.dim_model:]
        )
        #
        cell_t_new = gate_forget * self.cell_t + gate_input * gate_pre_c
        hidden_t_new = gate_output * numpy.tanh(cell_t_new)
        self.hidden_t = numpy.copy(hidden_t_new)
        self.cell_t = numpy.copy(cell_t_new)
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        #
        time_recent = self.one_seq[-1]['time_since_start']
        #
        delta = self.soft_relu(
            numpy.tensordot(
                self.hidden_t, self.W_delta, (0, 0)
            )
        )
        #
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde = numpy.sum(
            self.W_alpha * hidden_with_time,
            axis = 0
        ) + numpy.dot(
            self.hidden_t, self.W_mu
        )
        #
        self.intensity = self.soft_relu(
            self.intensity_tilde
        )
        # intensity computation is finished
    #
    def compute_intensity_upper_bound(self, time_current):
        # compute the upper bound of intensity
        # at the current time
        time_recent = self.one_seq[-1]['time_since_start']
        #
        delta = self.soft_relu(
            numpy.tensordot(
                self.hidden_t, self.W_delta, (0, 0)
            )
        )
        #
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde_ub = numpy.sum(
            self.hard_relu(
                self.W_alpha * hidden_with_time
            ),
            axis = 0
        ) + self.hard_relu(
            numpy.dot(
                self.hidden_t, self.W_mu
            )
        )
        #
        self.intensity_ub = self.soft_relu(
            self.intensity_tilde_ub
        )
        # intensity computation is finished
    #
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        #
        intensity_hazard = numpy.copy(
            self.intensity_ub[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            time_current += E / intensity_hazard
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
        #
        return time_current
        #
    #
    #
    def gen_one_seq(self, max_len):
        self.restart_sequence()
        '''
        Liiniger (2009), p. 28, describes a "thinning algorithm":
        generate one event of each type, take the minimum,
        and discard the others.
        Details found in my paper write-up
        #
        max_len is a pre-sampled value to set the length of seq
        '''
        # initialize the seq
        time_since_start = numpy.float32(0.0)
        time_since_start_each_event = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        #
        for idx_event in range(max_len):
            time_of_happen = numpy.zeros(
                (self.dim_process,), dtype=dtype
            )
            #
            # compute the hidden states
            # of the most recent event in sequence
            self.compute_hidden_states()
            #
            for type_event in range(self.dim_process):
                # sample one event using "thinning algorithm"
                time_of_happen[type_event] = numpy.copy(
                    self.sample_time_given_type(
                        type_event
                    )
                )
            #
            time_since_start_new = numpy.min(time_of_happen)
            type_event_new = numpy.argmin(time_of_happen)
            self.cnt_total_event += 1
            #
            # update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[type_event_new]
            time_since_start_each_event[type_event_new] = time_since_start
            self.one_seq.append(
                {
                    'idx_event': self.cnt_total_event,
                    'type_event': type_event_new,
                    'time_since_start': time_since_start,
                    'time_since_last_event': time_since_last_event,
                    'time_since_last_same_event': time_since_last_same_event
                }
            )
            #
        # throw away the BOS item
        # at the head of the sequence
        self.one_seq.pop(0)
        #
    #
    #
    def gen_seqs(self, settings):
        #
        #print settings
        num_seqs = settings['num_seqs']
        #
        self.list_seqs = []
        cnt_seqs = 0
        #for idx_seq in range(num_seqs):
        while cnt_seqs < num_seqs:
            #
            max_len = numpy.int32(
                round(
                    numpy.random.uniform(
                        low=settings['min_len'],
                        high=settings['max_len']
                    )
                )
            )
            #
            self.gen_one_seq(max_len)
            self.list_seqs.append(self.one_seq)
            cnt_seqs += 1
            if cnt_seqs % 10 == 9:
                print "idx seq of gen : ", (cnt_seqs, self.name)
                print "total number of seqs : ", num_seqs
    #
    #
    def print_some(self):
        print "printing some seqs ... "
        for idx_seq in range(10):
            print "the id of this seq is : ", idx_seq
            seq = self.list_seqs[idx_seq]
            list_events, list_time = [], []
            for event_item in seq:
                list_events.append(event_item['type_event'])
                list_time.append(
                    round(event_item['time_since_start'], 4)
                )
            print list_events
            print list_time
    #
    def save_seqs(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_seqs, f)


class NeuralHawkesAdaptiveBaseGen_time(object):
    '''
    here is the sequence generator using Neural Hawkes process
    '''
    def __init__(self, settings):
        #
        print "initializing generator ... "
        self.args = settings['args']
        self.dim_float = numpy.int32(32)
        if settings['path_pre_train'] == None:
            print "random parameters ... "
            self.dim_process = settings['dim_process']
            self.dim_model = settings['dim_LSTM']
            #
            self.dim_time = self.dim_float
            numpy.random.seed(
                settings['seed_random']
            )
            self.W_mu = numpy.float32(
                numpy.random.uniform(
                    low = -1.0, high = 1.0,
                    size = (
                        self.dim_model, self.dim_process
                    )
                )
            )
            #
            self.W_delta = numpy.float32(
                numpy.random.uniform(
                    low = -1.0, high = 1.0,
                    size=(
                        self.dim_model, self.dim_model,
                        self.dim_process
                    )
                )
            )
            #
            self.W_alpha = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.dim_model, self.dim_process)
                )
            )
            self.Emb_event = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        self.dim_process + numpy.int32(1),
                        self.dim_model
                    )
                )
            )
            self.Emb_time = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        self.dim_time+numpy.int32(1),
                        self.dim_model
                    )
                )
            )
            self.Threshold_time = numpy.float32(
                numpy.random.uniform(
                    low = 0.0, high = 1.0,
                    size = (self.dim_time, )
                )
            )
            self.W_recur = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        3 * self.dim_model,
                        4 * self.dim_model
                    )
                )
            )
            self.b_recur = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (4*self.dim_model, )
                )
            )
        else:
            print "read pretrained model ... "
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.W_mu = model_pre_train['W_mu']
            self.W_delta = model_pre_train['W_delta']
            self.W_alpha = model_pre_train['W_alpha']
            self.Emb_event = model_pre_train['Emb_event']
            self.Emb_time = model_pre_train['Emb_time']
            self.Threshold_time = model_pre_train['Threshold_time']
            self.W_recur = model_pre_train['W_recur']
            self.b_recur = model_pre_train['b_recur']
            #
        #
        #self.intensity = numpy.copy(self.mu)
        self.name = 'AdaptiveNeuralHawkesGen_time'
        #
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        print "initialization done "
        #
    #
    def soft_relu(self, x):
        return numpy.log(numpy.float32(1.0)+numpy.exp(x))
    #
    def hard_relu(self, x):
        return numpy.float32(0.5) * (x + numpy.abs(x) )
        #
    #
    def save_model(self, file_save):
        print "saving model of generator ... "
        model_dict = {
            'W_mu': numpy.copy(self.W_mu),
            'W_delta': numpy.copy(self.W_delta),
            'W_alpha': numpy.copy(self.W_alpha),
            'Emb_event': numpy.copy(self.Emb_event),
            'Emb_time': numpy.copy(self.Emb_time),
            'W_recur': numpy.copy(self.W_recur),
            'b_recur': numpy.copy(self.b_recur),
            'dim_process': self.dim_process,
            'dim_model': self.dim_model,
            'dim_time': self.dim_time,
            'dim_float': self.dim_float,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        #
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    #
    #
    def float32_to_bit(self, float_input):
        '''
        input a number in float, convert it to float32
        get its 32-bit representations
        '''
        float32_input = numpy.float32(float_input)
        str_input = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', float32_input))
        bit_input = numpy.zeros(
            (self.dim_float,), dtype=dtype
        )
        assert(self.dim_float == len(str_input))
        for idx, item_in_input in enumerate(str_input):
            bit_input[idx] = numpy.float32(item_in_input)
        return numpy.copy(bit_input)
    #
    #
    def sigmoid(self, x):
        return 1 / (1+numpy.exp(-x))
    #
    #
    def compute_hidden_states(self):
        # every time it is called,
        # it computes the new hidden states of the LSTM
        # it gets the last event in the sequence
        # which is generated at t_(rec(t))
        # and compute its hidden states
        emb_event_t = self.Emb_event[
            self.one_seq[-1]['type_event'], :
        ]
        #
        time_rep_t = self.hard_relu(
            self.one_seq[-1]['time_since_last_event'] - self.Threshold_time
        )
        time_rep_t = numpy.concatenate(
            (
                time_rep_t,
                self.one_seq[-1][
                    'time_since_last_event'
                ][None]
            ), axis = 0
        )
        emb_time_t = numpy.dot(
            time_rep_t, self.Emb_time
        )
        #
        post_transform = numpy.dot(
            numpy.concatenate(
                (emb_event_t, emb_time_t, self.hidden_t),
                axis = 0
            ),
            self.W_recur
        ) + self.b_recur
        #
        gate_input = self.sigmoid(
            post_transform[:self.dim_model]
        )
        gate_forget = self.sigmoid(
            post_transform[self.dim_model:2*self.dim_model]
        )
        gate_output = self.sigmoid(
            post_transform[2*self.dim_model:3*self.dim_model]
        )
        gate_pre_c = numpy.tanh(
            post_transform[3*self.dim_model:]
        )
        #
        cell_t_new = gate_forget * self.cell_t + gate_input * gate_pre_c
        hidden_t_new = gate_output * numpy.tanh(cell_t_new)
        self.hidden_t = numpy.copy(hidden_t_new)
        self.cell_t = numpy.copy(cell_t_new)
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        #
        time_recent = self.one_seq[-1]['time_since_start']
        #
        delta = self.soft_relu(
            numpy.tensordot(
                self.hidden_t, self.W_delta, (0, 0)
            )
        )
        #
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde = numpy.sum(
            self.W_alpha * hidden_with_time,
            axis = 0
        ) + numpy.dot(
            self.hidden_t, self.W_mu
        )
        #
        self.intensity = self.soft_relu(
            self.intensity_tilde
        )
        # intensity computation is finished
    #
    def compute_intensity_upper_bound(self, time_current):
        # compute the upper bound of intensity
        # at the current time
        time_recent = self.one_seq[-1]['time_since_start']
        #
        delta = self.soft_relu(
            numpy.tensordot(
                self.hidden_t, self.W_delta, (0, 0)
            )
        )
        #
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde_ub = numpy.sum(
            self.hard_relu(
                self.W_alpha * hidden_with_time
            ),
            axis = 0
        ) + numpy.dot(
            self.hidden_t, self.W_mu
        )
        # this part is time-invariant so
        # we do not need to take its hard_relu
        #self.hard_relu(
        #    numpy.dot(
        #        self.hidden_t, self.W_mu
        #    )
        #)
        #
        self.intensity_ub = self.soft_relu(
            self.intensity_tilde_ub
        )
        # intensity computation is finished
    #
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        #
        intensity_hazard = numpy.copy(
            self.intensity_ub[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            #print "type is : ", type_event
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            #print "E U time_current : "
            #print E, U, time_current
            #print "intensity hazard is : "
            #print intensity_hazard
            time_current += (E / intensity_hazard)
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
            #print "new time_current and u : "
            #print time_current, u
            #print "intensity and upper bound is : "
            #print self.intensity
            #print self.intensity_ub
            # use adaptive thinning algorithm
            # that is, decreasing the upper bound
            # to make the sampling quicker
            self.compute_intensity_upper_bound(
                time_current
            )
            intensity_hazard = numpy.copy(
                self.intensity_ub[type_event]
            )
        #
        return time_current
        #
    #
    #
    def gen_one_seq(self, max_len):
        self.restart_sequence()
        '''
        Liiniger (2009), p. 28, describes a "thinning algorithm":
        generate one event of each type, take the minimum,
        and discard the others.
        Details found in my paper write-up
        #
        max_len is a pre-sampled value to set the length of seq
        '''
        # initialize the seq
        time_since_start = numpy.float32(0.0)
        time_since_start_each_event = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        #
        for idx_event in range(max_len):
            time_of_happen = numpy.zeros(
                (self.dim_process,), dtype=dtype
            )
            #
            # compute the hidden states
            # of the most recent event in sequence
            self.compute_hidden_states()
            #
            for type_event in range(self.dim_process):
                # sample one event using "thinning algorithm"
                time_of_happen[type_event] = numpy.copy(
                    self.sample_time_given_type(
                        type_event
                    )
                )
            #
            time_since_start_new = numpy.min(time_of_happen)
            type_event_new = numpy.argmin(time_of_happen)
            self.cnt_total_event += 1
            #
            # update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[type_event_new]
            time_since_start_each_event[type_event_new] = time_since_start
            self.one_seq.append(
                {
                    'idx_event': self.cnt_total_event,
                    'type_event': type_event_new,
                    'time_since_start': time_since_start,
                    'time_since_last_event': time_since_last_event,
                    'time_since_last_same_event': time_since_last_same_event
                }
            )
            #
        # throw away the BOS item
        # at the head of the sequence
        self.one_seq.pop(0)
        #
    #
    #
    def gen_seqs(self, settings):
        #
        #print settings
        print "generating sequences ... "
        num_seqs = settings['num_seqs']
        #
        self.list_seqs = []
        cnt_seqs = 0
        #for idx_seq in range(num_seqs):
        while cnt_seqs < num_seqs:
            #
            max_len = numpy.int32(
                round(
                    numpy.random.uniform(
                        low=settings['min_len'],
                        high=settings['max_len']
                    )
                )
            )
            #
            self.gen_one_seq(max_len)
            self.list_seqs.append(self.one_seq)
            cnt_seqs += 1
            if cnt_seqs % 10 == 9:
                print "idx seq of gen : ", (cnt_seqs, self.name)
                print "total number of seqs : ", num_seqs
    #
    #
    def print_some(self):
        print "printing some seqs ... "
        for idx_seq in range(10):
            print "the id of this seq is : ", idx_seq
            seq = self.list_seqs[idx_seq]
            list_events = []
            list_time = []
            list_dtime = []
            list_items = []
            for event_item in seq:
                list_events.append(event_item['type_event'])
                list_time.append(
                    round(event_item['time_since_start'], 4)
                )
                list_dtime.append(
                    round(event_item['time_since_last_event'], 4)
                )
                list_items.append(
                    (
                        event_item['type_event'],
                        round(
                            event_item['time_since_last_event'], 4
                        )
                    )
                )
            print "the events, time and diff time for : ", idx_seq
            print list_events
            print list_time
            print list_dtime
            print "the list of items is : "
            print list_items
    #
    #
    def save_seqs(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_seqs, f)



class NeuralHawkesAdaptiveBaseGen_time_scale(object):
    '''
    here is the sequence generator using Neural Hawkes process
    '''
    def __init__(self, settings):
        #
        print "initializing generator ... "
        self.args = settings['args']
        self.dim_float = numpy.int32(32)
        if settings['path_pre_train'] == None:
            print "random parameters ... "
            self.dim_process = settings['dim_process']
            self.dim_model = settings['dim_LSTM']
            self.dim_time = self.dim_float
            #
            numpy.random.seed(
                settings['seed_random']
            )
            #
            #self.scale = numpy.float32(
            #    numpy.random.uniform(
            #        low = 1e-3, high = 2.0,
            #        size = (self.dim_process, )
            #    )
            #)
            self.scale = numpy.float32(
                numpy.ones( (self.dim_process, ) )
            )
            #
            self.W_mu = numpy.float32(
                numpy.random.uniform(
                    low = -1.0, high = 1.0,
                    size = (
                        self.dim_model, self.dim_process
                    )
                )
            )
            #
            self.W_delta = numpy.float32(
                numpy.random.uniform(
                    low = -1.0, high = 1.0,
                    size=(
                        self.dim_model, self.dim_model,
                        self.dim_process
                    )
                )
            )
            #
            self.W_alpha = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.dim_model, self.dim_process)
                )
            )
            self.Emb_event = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        self.dim_process + numpy.int32(1),
                        self.dim_model
                    )
                )
            )
            self.Emb_time = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        self.dim_time+numpy.int32(1),
                        self.dim_model
                    )
                )
            )
            self.Threshold_time = numpy.float32(
                numpy.random.uniform(
                    low = 0.0, high = 1.0,
                    size = (self.dim_time, )
                )
            )
            self.W_recur = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        3 * self.dim_model,
                        4 * self.dim_model
                    )
                )
            )
            self.b_recur = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (4*self.dim_model, )
                )
            )
        else:
            print "read pretrained model ... "
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.scale = model_pre_train['scale']
            self.W_mu = model_pre_train['W_mu']
            self.W_delta = model_pre_train['W_delta']
            self.W_alpha = model_pre_train['W_alpha']
            self.Emb_event = model_pre_train['Emb_event']
            self.Emb_time = model_pre_train['Emb_time']
            self.Threshold_time = model_pre_train['Threshold_time']
            self.W_recur = model_pre_train['W_recur']
            self.b_recur = model_pre_train['b_recur']
            #
        #
        #self.intensity = numpy.copy(self.mu)
        self.name = 'AdaptiveNeuralHawkesGen_time_scale'
        #
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        print "initialization done "
        #
    #
    def soft_relu(self, x):
        return numpy.log(numpy.float32(1.0)+numpy.exp(x))
    #
    def soft_relu_scale(self, x):
        # last dim of x is dim_process
        x /= self.scale
        y = numpy.log(numpy.float32(1.0)+numpy.exp(x))
        y *= self.scale
        return y
    #
    def hard_relu(self, x):
        return numpy.float32(0.5) * (x + numpy.abs(x) )
        #
    #
    def save_model(self, file_save):
        print "saving model of generator ... "
        model_dict = {
            'scale': numpy.copy(self.scale),
            'W_mu': numpy.copy(self.W_mu),
            'W_delta': numpy.copy(self.W_delta),
            'W_alpha': numpy.copy(self.W_alpha),
            'Emb_event': numpy.copy(self.Emb_event),
            'Emb_time': numpy.copy(self.Emb_time),
            'Threshold_time': numpy.copy(self.Threshold_time),
            'W_recur': numpy.copy(self.W_recur),
            'b_recur': numpy.copy(self.b_recur),
            'dim_process': self.dim_process,
            'dim_model': self.dim_model,
            'dim_time': self.dim_time,
            'dim_float': self.dim_float,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        #
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    #
    #
    def float32_to_bit(self, float_input):
        '''
        input a number in float, convert it to float32
        get its 32-bit representations
        '''
        float32_input = numpy.float32(float_input)
        str_input = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', float32_input))
        bit_input = numpy.zeros(
            (self.dim_float,), dtype=dtype
        )
        assert(self.dim_float == len(str_input))
        for idx, item_in_input in enumerate(str_input):
            bit_input[idx] = numpy.float32(item_in_input)
        return numpy.copy(bit_input)
    #
    #
    def sigmoid(self, x):
        return 1 / (1+numpy.exp(-x))
    #
    #
    def compute_hidden_states(self):
        # every time it is called,
        # it computes the new hidden states of the LSTM
        # it gets the last event in the sequence
        # which is generated at t_(rec(t))
        # and compute its hidden states
        emb_event_t = self.Emb_event[
            self.one_seq[-1]['type_event'], :
        ]
        #
        time_rep_t = self.hard_relu(
            self.one_seq[-1]['time_since_last_event'] - self.Threshold_time
        )
        time_rep_t = numpy.concatenate(
            (
                time_rep_t,
                self.one_seq[-1][
                    'time_since_last_event'
                ][None]
            ), axis = 0
        )
        emb_time_t = numpy.dot(
            time_rep_t, self.Emb_time
        )
        #
        post_transform = numpy.dot(
            numpy.concatenate(
                (emb_event_t, emb_time_t, self.hidden_t),
                axis = 0
            ),
            self.W_recur
        ) + self.b_recur
        #
        gate_input = self.sigmoid(
            post_transform[:self.dim_model]
        )
        gate_forget = self.sigmoid(
            post_transform[self.dim_model:2*self.dim_model]
        )
        gate_output = self.sigmoid(
            post_transform[2*self.dim_model:3*self.dim_model]
        )
        gate_pre_c = numpy.tanh(
            post_transform[3*self.dim_model:]
        )
        #
        cell_t_new = gate_forget * self.cell_t + gate_input * gate_pre_c
        hidden_t_new = gate_output * numpy.tanh(cell_t_new)
        self.hidden_t = numpy.copy(hidden_t_new)
        self.cell_t = numpy.copy(cell_t_new)
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        #
        time_recent = self.one_seq[-1]['time_since_start']
        #
        delta = self.soft_relu(
            numpy.tensordot(
                self.hidden_t, self.W_delta, (0, 0)
            )
        )
        #
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde = numpy.sum(
            self.W_alpha * hidden_with_time,
            axis = 0
        ) + numpy.dot(
            self.hidden_t, self.W_mu
        )
        #
        self.intensity = self.soft_relu_scale(
            self.intensity_tilde
        )
        # intensity computation is finished
    #
    def compute_intensity_upper_bound(self, time_current):
        # compute the upper bound of intensity
        # at the current time
        time_recent = self.one_seq[-1]['time_since_start']
        #
        delta = self.soft_relu(
            numpy.tensordot(
                self.hidden_t, self.W_delta, (0, 0)
            )
        )
        #
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t[:, None]
        # (self.dim_model, self.dim_process)
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde_ub = numpy.sum(
            self.hard_relu(
                self.W_alpha * hidden_with_time
            ),
            axis = 0
        ) + numpy.dot(
            self.hidden_t, self.W_mu
        )
        # this part is time-invariant so
        # we do not need to take its hard_relu
        #self.hard_relu(
        #    numpy.dot(
        #        self.hidden_t, self.W_mu
        #    )
        #)
        #
        self.intensity_ub = self.soft_relu_scale(
            self.intensity_tilde_ub
        )
        # intensity computation is finished
    #
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        intensity_hazard = numpy.copy(
            self.intensity_ub[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            #print "type is : ", type_event
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            #print "E U time_current : "
            #print E, U, time_current
            #print "intensity hazard is : "
            #print intensity_hazard
            time_current += (E / intensity_hazard)
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
            #print "new time_current and u : "
            #print time_current, u
            #print "intensity and upper bound is : "
            #print self.intensity
            #print self.intensity_ub
            # use adaptive thinning algorithm
            # that is, decreasing the upper bound
            # to make the sampling quicker
            # use adaptive method by
            # toggling on the following block
            '''
            self.compute_intensity_upper_bound(
                time_current
            )
            intensity_hazard = numpy.copy(
                self.intensity_ub[type_event]
            )
            '''
        return time_current
        #
    #
    #
    def gen_one_seq(self, max_len):
        self.restart_sequence()
        '''
        Liiniger (2009), p. 28, describes a "thinning algorithm":
        generate one event of each type, take the minimum,
        and discard the others.
        Details found in my paper write-up
        max_len is a pre-sampled value to set the length of seq
        '''
        # initialize the seq
        time_since_start = numpy.float32(0.0)
        time_since_start_each_event = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        #
        for idx_event in range(max_len):
            time_of_happen = numpy.zeros(
                (self.dim_process,), dtype=dtype
            )
            #
            # compute the hidden states
            # of the most recent event in sequence
            self.compute_hidden_states()
            #
            for type_event in range(self.dim_process):
                # sample one event using "thinning algorithm"
                time_of_happen[type_event] = numpy.copy(
                    self.sample_time_given_type(
                        type_event
                    )
                )
            #
            time_since_start_new = numpy.min(time_of_happen)
            type_event_new = numpy.argmin(time_of_happen)
            self.cnt_total_event += 1
            #
            # update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[type_event_new]
            time_since_start_each_event[type_event_new] = time_since_start
            self.one_seq.append(
                {
                    'idx_event': self.cnt_total_event,
                    'type_event': type_event_new,
                    'time_since_start': time_since_start,
                    'time_since_last_event': time_since_last_event,
                    'time_since_last_same_event': time_since_last_same_event
                }
            )
            #
        # throw away the BOS item
        # at the head of the sequence
        self.one_seq.pop(0)
        #
    #
    #
    def gen_seqs(self, settings):
        #
        #print settings
        print "generating sequences ... "
        num_seqs = settings['num_seqs']
        #
        self.list_seqs = []
        cnt_seqs = 0
        #for idx_seq in range(num_seqs):
        while cnt_seqs < num_seqs:
            #
            max_len = numpy.int32(
                round(
                    numpy.random.uniform(
                        low=settings['min_len'],
                        high=settings['max_len']
                    )
                )
            )
            #
            self.gen_one_seq(max_len)
            self.list_seqs.append(self.one_seq)
            cnt_seqs += 1
            if cnt_seqs % 10 == 9:
                print "idx seq of gen : ", (cnt_seqs, self.name)
                print "total number of seqs : ", num_seqs
    #
    #
    def print_some(self):
        print "printing some seqs ... "
        for idx_seq in range(10):
            print "the id of this seq is : ", idx_seq
            seq = self.list_seqs[idx_seq]
            list_events = []
            list_time = []
            list_dtime = []
            list_items = []
            for event_item in seq:
                list_events.append(event_item['type_event'])
                list_time.append(
                    round(event_item['time_since_start'], 4)
                )
                list_dtime.append(
                    round(event_item['time_since_last_event'], 4)
                )
                list_items.append(
                    (
                        event_item['type_event'],
                        round(
                            event_item['time_since_last_event'], 4
                        )
                    )
                )
            print "the events, time and diff time for : ", idx_seq
            print list_events
            print list_time
            print list_dtime
            print "the list of items is : "
            print list_items
    #
    #
    def save_seqs(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_seqs, f)



class NeuralHawkesAdaptiveBaseGen_time_scale_reduce(
    object
):
    '''
    here is the sequence generator
    using Neural Hawkes process with reduced decay param
    '''
    def __init__(self, settings):
        #
        print "initializing generator ... "
        self.args = settings['args']
        self.dim_float = numpy.int32(32)
        if settings['path_pre_train'] == None:
            print "random parameters ... "
            self.dim_process = settings['dim_process']
            self.dim_model = settings['dim_LSTM']
            self.dim_time = self.dim_float
            #
            numpy.random.seed(
                settings['seed_random']
            )
            #
            #self.scale = numpy.float32(
            #    numpy.random.uniform(
            #        low = 1e-3, high = 2.0,
            #        size = (self.dim_process, )
            #    )
            #)
            self.scale = numpy.float32(
                numpy.ones( (self.dim_process, ) )
            )
            #
            self.W_mu = numpy.float32(
                numpy.random.uniform(
                    low = -1.0, high = 1.0,
                    size = (
                        self.dim_model, self.dim_process
                    )
                )
            )
            #
            self.W_delta = numpy.float32(
                numpy.random.uniform(
                    low = -1.0, high = 1.0,
                    size=(
                        self.dim_model,
                        self.dim_model
                    )
                )
            )
            #
            self.W_alpha = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.dim_model, self.dim_process)
                )
            )
            self.Emb_event = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        self.dim_process + numpy.int32(1),
                        self.dim_model
                    )
                )
            )
            self.Emb_time = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        self.dim_time+numpy.int32(1),
                        self.dim_model
                    )
                )
            )
            self.Threshold_time = numpy.float32(
                numpy.random.uniform(
                    low = 0.0, high = 1.0,
                    size = (self.dim_time, )
                )
            )
            self.W_recur = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (
                        3 * self.dim_model,
                        4 * self.dim_model
                    )
                )
            )
            self.b_recur = numpy.float32(
                numpy.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (4*self.dim_model, )
                )
            )
        else:
            print "read pretrained model ... "
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.scale = model_pre_train['scale']
            self.W_mu = model_pre_train['W_mu']
            self.W_delta = model_pre_train['W_delta']
            self.W_alpha = model_pre_train['W_alpha']
            self.Emb_event = model_pre_train['Emb_event']
            self.Emb_time = model_pre_train['Emb_time']
            self.Threshold_time = model_pre_train['Threshold_time']
            self.W_recur = model_pre_train['W_recur']
            self.b_recur = model_pre_train['b_recur']
            #
        #
        #self.intensity = numpy.copy(self.mu)
        self.name = 'AdaptiveNeuralHawkesGen_time_scale_reduce'
        #
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq) )
        print "initialization done "
        #
    #
    def soft_relu(self, x):
        return numpy.log(numpy.float32(1.0)+numpy.exp(x))
    #
    def soft_relu_scale(self, x):
        # last dim of x is dim_process
        x /= self.scale
        y = numpy.log(numpy.float32(1.0)+numpy.exp(x))
        y *= self.scale
        return y
    #
    def hard_relu(self, x):
        return numpy.float32(0.5) * (x + numpy.abs(x) )
        #
    #
    def save_model(self, file_save):
        print "saving model of generator ... "
        model_dict = {
            'scale': numpy.copy(self.scale),
            'W_mu': numpy.copy(self.W_mu),
            'W_delta': numpy.copy(self.W_delta),
            'W_alpha': numpy.copy(self.W_alpha),
            'Emb_event': numpy.copy(self.Emb_event),
            'Emb_time': numpy.copy(self.Emb_time),
            'Threshold_time': numpy.copy(self.Threshold_time),
            'W_recur': numpy.copy(self.W_recur),
            'b_recur': numpy.copy(self.b_recur),
            'dim_process': self.dim_process,
            'dim_model': self.dim_model,
            'dim_time': self.dim_time,
            'dim_float': self.dim_float,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    #
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        #
        self.one_seq.append(
            {
                'idx_event': numpy.int32(0),
                'type_event': self.dim_process,
                'time_since_start': numpy.float32(0.0),
                'time_since_last_event': numpy.float32(0.0),
                'time_since_last_same_event': numpy.float32(0.0)
            }
        )
        self.hidden_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq))
        #
    #
    #
    #
    def float32_to_bit(self, float_input):
        '''
        input a number in float, convert it to float32
        get its 32-bit representations
        '''
        float32_input = numpy.float32(float_input)
        str_input = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', float32_input))
        bit_input = numpy.zeros(
            (self.dim_float,), dtype=dtype
        )
        assert(self.dim_float == len(str_input))
        for idx, item_in_input in enumerate(str_input):
            bit_input[idx] = numpy.float32(item_in_input)
        return numpy.copy(bit_input)
    #
    #
    def sigmoid(self, x):
        return 1 / (1+numpy.exp(-x))
    #
    #
    def compute_hidden_states(self):
        # every time it is called,
        # it computes the new hidden states of the LSTM
        # it gets the last event in the sequence
        # which is generated at t_(rec(t))
        # and compute its hidden states
        emb_event_t = self.Emb_event[
            self.one_seq[-1]['type_event'], :
        ]
        #
        time_rep_t = self.hard_relu(
            self.one_seq[-1]['time_since_last_event'] - self.Threshold_time
        )
        time_rep_t = numpy.concatenate(
            (
                time_rep_t,
                self.one_seq[-1][
                    'time_since_last_event'
                ][None]
            ), axis = 0
        )
        emb_time_t = numpy.dot(
            time_rep_t, self.Emb_time
        )
        #
        post_transform = numpy.dot(
            numpy.concatenate(
                (emb_event_t, emb_time_t, self.hidden_t),
                axis = 0
            ),
            self.W_recur
        ) + self.b_recur
        #
        gate_input = self.sigmoid(
            post_transform[:self.dim_model]
        )
        gate_forget = self.sigmoid(
            post_transform[self.dim_model:2*self.dim_model]
        )
        gate_output = self.sigmoid(
            post_transform[2*self.dim_model:3*self.dim_model]
        )
        gate_pre_c = numpy.tanh(
            post_transform[3*self.dim_model:]
        )
        #
        cell_t_new = gate_forget * self.cell_t + gate_input * gate_pre_c
        hidden_t_new = gate_output * numpy.tanh(cell_t_new)
        self.hidden_t = numpy.copy(hidden_t_new)
        self.cell_t = numpy.copy(cell_t_new)
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        time_recent = self.one_seq[-1]['time_since_start']
        # W_delta : dim_model * dim_model
        delta = self.soft_relu(
            numpy.dot(
                self.hidden_t, self.W_delta
            )
        )
        # dim_model
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t
        # dim_model
        # self.W_alpha (self.dim_model, self.dim_process)
        self.intensity_tilde = numpy.dot(
            hidden_with_time, self.W_alpha
        ) + numpy.dot(
            self.hidden_t, self.W_mu
        )
        #
        self.intensity = self.soft_relu_scale(
            self.intensity_tilde
        )
        # intensity computation is finished
    #
    #
    def compute_intensity_upper_bound(self, time_current):
        # compute the upper bound of intensity
        # at the current time
        time_recent = self.one_seq[-1]['time_since_start']
        #
        delta = self.soft_relu(
            numpy.dot(
                self.hidden_t, self.W_delta
            )
        )
        #
        hidden_with_time = numpy.exp(
            -delta * (
                time_current - time_recent
            )
        ) * self.hidden_t
        # hidden_with_time : dim_model
        self.intensity_tilde_ub = numpy.sum(
            self.hard_relu(
                self.W_alpha * hidden_with_time[:, None]
            ),
            axis = 0
        ) + numpy.dot(
            self.hidden_t, self.W_mu
        )
        # this part is time-invariant so
        # we do not need to take its hard_relu
        #self.hard_relu(
        #    numpy.dot(
        #        self.hidden_t, self.W_mu
        #    )
        #)
        #
        self.intensity_ub = self.soft_relu_scale(
            self.intensity_tilde_ub
        )
        # intensity computation is finished
    #
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        intensity_hazard = numpy.copy(
            self.intensity_ub[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            #print "type is : ", type_event
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            #print "E U time_current : "
            #print E, U, time_current
            #print "intensity hazard is : "
            #print intensity_hazard
            time_current += (E / intensity_hazard)
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
            #print "new time_current and u : "
            #print time_current, u
            #print "intensity and upper bound is : "
            #print self.intensity
            #print self.intensity_ub
            # use adaptive thinning algorithm
            # that is, decreasing the upper bound
            # to make the sampling quicker
            # use adaptive method by
            # toggling on the following block
            '''
            self.compute_intensity_upper_bound(
                time_current
            )
            intensity_hazard = numpy.copy(
                self.intensity_ub[type_event]
            )
            '''
        return time_current
        #
    #
    #
    def gen_one_seq(self, max_len):
        self.restart_sequence()
        '''
        Liiniger (2009), p. 28, describes a "thinning algorithm":
        generate one event of each type, take the minimum,
        and discard the others.
        Details found in my paper write-up
        max_len is a pre-sampled value to set the length of seq
        '''
        # initialize the seq
        time_since_start = numpy.float32(0.0)
        time_since_start_each_event = numpy.zeros(
            (self.dim_process,), dtype=dtype
        )
        #
        for idx_event in range(max_len):
            time_of_happen = numpy.zeros(
                (self.dim_process,), dtype=dtype
            )
            #
            # compute the hidden states
            # of the most recent event in sequence
            self.compute_hidden_states()
            #
            for type_event in range(self.dim_process):
                # sample one event using "thinning algorithm"
                time_of_happen[type_event] = numpy.copy(
                    self.sample_time_given_type(
                        type_event
                    )
                )
            #
            time_since_start_new = numpy.min(time_of_happen)
            type_event_new = numpy.argmin(time_of_happen)
            self.cnt_total_event += 1
            #
            # update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[type_event_new]
            time_since_start_each_event[type_event_new] = time_since_start
            self.one_seq.append(
                {
                    'idx_event': self.cnt_total_event,
                    'type_event': type_event_new,
                    'time_since_start': time_since_start,
                    'time_since_last_event': time_since_last_event,
                    'time_since_last_same_event': time_since_last_same_event
                }
            )
            #
        # throw away the BOS item
        # at the head of the sequence
        self.one_seq.pop(0)
        #
    #
    #
    def gen_seqs(self, settings):
        #
        #print settings
        print "generating sequences ... "
        num_seqs = settings['num_seqs']
        #
        self.list_seqs = []
        cnt_seqs = 0
        #for idx_seq in range(num_seqs):
        while cnt_seqs < num_seqs:
            #
            max_len = numpy.int32(
                round(
                    numpy.random.uniform(
                        low=settings['min_len'],
                        high=settings['max_len']
                    )
                )
            )
            #
            self.gen_one_seq(max_len)
            self.list_seqs.append(self.one_seq)
            cnt_seqs += 1
            if cnt_seqs % 10 == 9:
                print "idx seq of gen : ", (cnt_seqs, self.name)
                print "total number of seqs : ", num_seqs
    #
    #
    def print_some(self):
        print "printing some seqs ... "
        for idx_seq in range(10):
            print "the id of this seq is : ", idx_seq
            seq = self.list_seqs[idx_seq]
            list_events = []
            list_time = []
            list_dtime = []
            list_items = []
            for event_item in seq:
                list_events.append(event_item['type_event'])
                list_time.append(
                    round(event_item['time_since_start'], 4)
                )
                list_dtime.append(
                    round(event_item['time_since_last_event'], 4)
                )
                list_items.append(
                    (
                        event_item['type_event'],
                        round(
                            event_item['time_since_last_event'], 4
                        )
                    )
                )
            print "the events, time and diff time for : ", idx_seq
            print list_events
            print list_time
            print list_dtime
            print "the list of items is : "
            print list_items
    #
    #
    def save_seqs(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_seqs, f)



#
#
