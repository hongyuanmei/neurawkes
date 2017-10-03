# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8th 10:58:37 2016

data processers

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
import struct

import utils


dtype=theano.config.floatX

class DataProcesser(object):
    '''
    this class process raw data into the model-friendly format
    and save them when neccessary
    '''
    def __init__(self, settings):
        #
        print "initialize the data processer ... "
        self.path_rawdata = os.path.abspath(
            settings['path_rawdata']
        )
        self.size_batch = settings['size_batch']
        # hard code dimensions related to time
        self.dim_float = numpy.int32(32)
        self.dim_time = self.dim_float
        # dim_time might be changed when we consider more features
        #
        self.to_read = settings['to_read']
        self.ratio_train = numpy.float32(settings['ratio_train'])
        self.data = {
            'train': [],
            'dev': [], 'devtest': [], 'test': []
        }
        for tag_split in self.to_read:
            print "reading data for tag : ", tag_split
            path_to_read = self.path_rawdata + '/' + tag_split + '.pkl'
            with open(path_to_read, 'rb') as f:
                data_temp = pickle.load(f)
                self.data[tag_split] = data_temp[tag_split]
                if 'dim_process' in data_temp:
                    self.data['dim_process'] = data_temp['dim_process']
        #
        #
        if 'dim_process' in self.data:
            # if the dim_process is given by the dataset, like food diery
            self.dim_process = numpy.int32(
                self.data['dim_process']
            )
        else:
            # if it is not given, like in simulated data and in retweet data
            counts = {}
            for seq in self.data['train']:
                for item in seq:
                    type_event = item['type_event']
                    if type_event in counts:
                        counts[type_event] += 1
                    else:
                        counts[type_event] = 1
            self.dim_process = numpy.int32(len(counts) )
        #
        len_to_select = numpy.int32(
            len(self.data['train']) * self.ratio_train
        )
        self.data['train'] = self.data['train'][:len_to_select]
        #
        self.lens = {
            'train': len(self.data['train']),
            'dev': len(self.data['dev']),
            'devtest': len(self.data['devtest']),
            'test': len(self.data['test'])
        }
        self.list_idx = {
            'train': range(self.lens['train']),
            'dev': range(self.lens['dev']),
            'devtest': range(self.lens['devtest']),
            'test': range(self.lens['test'])
        }
        # chop the data
        # computing max num of batches for all data ...
        self.max_nums = {
            'train': int( self.lens['train']/self.size_batch ),
            'dev': int( self.lens['dev']/self.size_batch ),
            'devtest': int( self.lens['devtest']/self.size_batch ),
            'test': int( self.lens['test']/self.size_batch )
        }
        #
        # decide if we only predict partial stream
        self.partial_predict = settings['partial_predict']
        self.substream = [0]
        '''
        for now, partial predict is only to predict event-0
        in the future, it can be more general
        like predicting a substream 0, 2, 4, ...
        this feature is useless now ...
        '''
        #
        print "finish data processer initialization ... "
        #
        #
        #
    #
    #
    def re_init(self):
        #
        print "re-initialize the data processer for DevIncludedSetting ... "
        #
        data_temp = self.data
        self.data = {
            'train': [], 'dev': [],
            'devtest': [], 'test': []
        }
        self.data['train'] = data_temp['train'] + data_temp['dev']
        self.data['dev'] = data_temp['test']
        #
        self.lens = {
            'train': len(self.data['train']),
            'dev': len(self.data['dev']),
            'devtest': len(self.data['devtest']),
            'test': len(self.data['test'])
        }
        print "lens : ", self.lens
        self.list_idx = {
            'train': range(self.lens['train']),
            'dev': range(self.lens['dev']),
            'devtest': range(self.lens['devtest']),
            'test': range(self.lens['test'])
        }
        # chop the data
        # computing max num of batches for all data ...
        self.max_nums = {
            'train': int( self.lens['train']/self.size_batch ),
            'dev': int( self.lens['dev']/self.size_batch ),
            'devtest': int( self.lens['devtest']/self.size_batch ),
            'test': int( self.lens['test']/self.size_batch )
        }
        #
        print "finish data processer re-initialization for DevIncludedSetting ... "
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
    def get_time_quantiles(self):
        self.time_quantiles = numpy.zeros(
            (self.dim_time, ), dtype=dtype
        )
        list_times = []
        for seq in self.data['train']:
            for item in seq:
                list_times.append(
                    item['time_since_last_event']
                )
        list_times = sorted(list_times)
        list_len = len(list_times)
        for i in range(self.dim_time):
            index = numpy.int32(list_len*i/self.dim_time)
            self.time_quantiles[i] = numpy.float32(list_times[index])
        self.time_quantiles[0] = 0.0
    #

    def shuffle_train_data(self):
        #assert(tag=='train')
        print "shuffling training data idx ... "
        # we shuffle idx instead of the real data
        numpy.random.shuffle(self.list_idx['train'])

    #
    def prune_stream(self, tag_discard):
        #TODO: prune the stream by discarding events
        # tag_discard can be :
        # 0, 1, 2, ..., 2**dim_process-1
        print "pruning stream ... "
        print "find the substream to keep ... "
        tag_discard = numpy.int32(tag_discard)
        assert(
            tag_discard >= 0 and tag_discard <= 2**self.dim_process-1
        )
        bin_str = format(
            tag_discard, 'b'
        ).zfill(self.dim_process)
        self.list_to_keep = []
        for idx_process in range(self.dim_process):
            if bin_str[self.dim_process-idx_process-1] == '0':
                self.list_to_keep.append(idx_process)
        #
        print "discarding except : ", self.list_to_keep
        data_temp = self.data
        #print "keys of data_temp is : ", data_temp.keys()
        self.data = {}
        # ['test', 'dim_process', 'train', 'devtest', 'dev']
        for tag_split in ['train', 'dev', 'devtest', 'test']:
            self.data[tag_split] = []
            for seq in data_temp[tag_split]:
                self.data[tag_split].append(
                    [
                        item for item in seq if item['type_event'] in self.list_to_keep
                    ]
                )
        print "re-compute the time since last event ... "
        for tag_split in ['train', 'dev', 'devtest', 'test']:
            for seq in self.data[tag_split]:
                if len(seq) >= 1:
                    seq[0]['time_since_last_event'] = seq[0]['time_since_start']
                if len(seq) >= 2:
                    for idx, item in enumerate(seq[1:]):
                        item['time_since_last_event'] = item['time_since_start'] - seq[idx]['time_since_start']
        #
    #

    def process_seq(self):
        #print "getting batch ... "
        max_len = 0
        temp_list_seq_type_event = []
        temp_list_seq_time_since_last = []
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            temp_seq_type_event = []
            temp_seq_time_since_last = []
            for item in self.data[self.tag_batch][idx_data]:
                temp_seq_type_event.append(
                    item['type_event']
                )
                temp_seq_time_since_last.append(
                    item['time_since_last_event']
                )
            len_seq = len(temp_seq_type_event)
            if max_len < len_seq:
                max_len = len_seq
            temp_list_seq_type_event.append(
                temp_seq_type_event
            )
            temp_list_seq_time_since_last.append(
                temp_seq_time_since_last
            )
        #
        self.seq_type_event_numpy = numpy.zeros(
            (max_len, self.size_batch), dtype=numpy.int32
        )
        self.seq_time_since_last_numpy = numpy.zeros(
            (max_len, self.size_batch), dtype=dtype
        )
        self.seq_mask_numpy = numpy.zeros(
            (max_len, self.size_batch), dtype=dtype
        )
        #
        for idx_in_batch, (seq_type_event, seq_time_since_last) in enumerate(zip(temp_list_seq_type_event, temp_list_seq_time_since_last)):
            for idx_pos, (type_event, time_since_start) in enumerate(zip(seq_type_event, seq_time_since_last)):
                self.seq_type_event_numpy[idx_pos, idx_in_batch] = type_event
                self.seq_time_since_last_numpy[idx_pos, idx_in_batch] = time_since_start
                self.seq_mask_numpy[idx_pos, idx_in_batch] = numpy.float32(1.0)
        #
        #

    def process_seq_hawkes(self, predict_first):
        #print "getting batch ... "
        self.max_len = 0
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            len_seq = len(
                self.data[self.tag_batch][idx_data]
            )
            if self.max_len < len_seq:
                self.max_len = len_seq
        #
        if self.max_len < 1:
            self.max_len = numpy.int32(1)
        #
        self.seq_time_to_end_numpy = numpy.zeros(
            (self.max_len, self.size_batch), dtype=dtype
        )
        self.seq_time_to_current_numpy = numpy.zeros(
            (self.max_len, self.max_len, self.size_batch), dtype=dtype
        )
        self.seq_type_event_numpy = numpy.zeros(
            (self.max_len, self.size_batch), dtype=numpy.int32
        )
        self.time_since_start_to_end_numpy = numpy.zeros(
            (self.size_batch,), dtype=dtype
        )
        self.seq_mask_numpy = numpy.zeros(
            (self.max_len, self.size_batch), dtype=dtype
        )
        self.seq_mask_to_current_numpy = numpy.zeros(
            (self.max_len, self.max_len, self.size_batch), dtype=dtype
        )
        #
        self.num_events_start_to_end_numpy = numpy.zeros(
            (self.size_batch, ), dtype=dtype
        )
        #self.num_sims_start_to_end = numpy.zeros(
        #    (self.size_batch, ), dtype=dtype
        #)
        #
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            seq = self.data[self.tag_batch][idx_data]
            #
            if len(seq) > 0:
                time_end = seq[-1]['time_since_start']
                #
                self.time_since_start_to_end_numpy[
                    idx_in_batch
                ] = time_end
                #
                self.num_events_start_to_end_numpy[
                    idx_in_batch
                ] = numpy.float32(len(seq) )
            else:
                time_end = numpy.float32(0.0)
                self.time_since_start_to_end_numpy[
                    idx_in_batch
                ] = time_end
                self.num_events_start_to_end_numpy[
                    idx_in_batch
                ] = numpy.float32(1)
            #self.num_sims_start_to_end[
            #    idx_in_batch
            #] = numpy.float32(
            #    multiple * self.num_events_start_to_end[idx_in_batch]
            #)
            #
            for idx_pos, item_event in enumerate(seq):
                #
                t_i = item_event['time_since_start']
                #
                self.seq_time_to_end_numpy[
                    idx_pos, idx_in_batch
                ] = time_end - t_i
                self.seq_type_event_numpy[
                    idx_pos, idx_in_batch
                ] = item_event['type_event']
                self.seq_mask_numpy[
                    idx_pos, idx_in_batch
                ] = numpy.float32(1.0)
                #
                if self.partial_predict:
                    if item_event['type_event'] not in self.substream:
                        self.seq_mask_numpy[idx_pos, idx_in_batch] = numpy.float32(0.0)
                #
                idx_pos_prime = 0
                while idx_pos_prime < idx_pos:
                    #
                    item_event_prime = seq[idx_pos_prime]
                    t_i_prime = item_event_prime[
                        'time_since_start'
                    ]
                    self.seq_time_to_current_numpy[
                        idx_pos, idx_pos_prime, idx_in_batch
                    ] = t_i - t_i_prime
                    self.seq_mask_to_current_numpy[
                        idx_pos, idx_pos_prime, idx_in_batch
                    ] = numpy.float32(1.0)
                    #
                    idx_pos_prime += 1
                    #
        if not predict_first:
            self.seq_mask_numpy[0,:] = numpy.float32(0.0)
        #
        #
    #
    #
    def sample_times(self, multiple=numpy.int32(10) ):
        self.num_sims_start_to_end_numpy = numpy.float32(
            multiple * self.num_events_start_to_end_numpy
        )
        self.max_num_sims = numpy.int32(
            numpy.max(
                self.num_sims_start_to_end_numpy
            )
        )
        self.seq_sims_time_to_current_numpy = numpy.zeros(
            (self.max_num_sims, self.max_len, self.size_batch), dtype=dtype
        )
        self.seq_sims_mask_to_current_numpy = numpy.zeros(
            (self.max_num_sims, self.max_len, self.size_batch), dtype=dtype
        )
        self.seq_sims_mask_numpy = numpy.zeros(
            (self.max_num_sims, self.size_batch),
            dtype=dtype
        )
        #
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            seq = self.data[self.tag_batch][idx_data]
            num_sims_this_seq = numpy.int32(
                self.num_sims_start_to_end_numpy[idx_in_batch]
            )
            #
            self.seq_sims_mask_numpy[
                :num_sims_this_seq, idx_in_batch
            ] = numpy.float32(1.0)
            #
            sampled_times = numpy.sort(
                numpy.random.uniform(
                    low = 0.0,
                    high = self.time_since_start_to_end_numpy[idx_in_batch],
                    size = (num_sims_this_seq,)
                )
            )
            temp_column = numpy.zeros(
                (self.max_num_sims,), dtype=dtype
            )
            temp_column[:num_sims_this_seq] = numpy.copy(
                sampled_times
            )
            for idx_pos_prime, item_event_prime in enumerate(seq):
                t_i_prime = item_event_prime[
                    'time_since_start'
                ]
                indices_to_edit = (temp_column > t_i_prime)
                self.seq_sims_time_to_current_numpy[
                    indices_to_edit, idx_pos_prime, idx_in_batch
                ] = (temp_column - t_i_prime)[indices_to_edit]
                self.seq_sims_mask_to_current_numpy[
                    indices_to_edit, idx_pos_prime, idx_in_batch
                ] = numpy.float32(1.0)

        #
        #
        #
    #
    #
    #
    def get_tensors_hawkes_given_sampled_times(self):
        #
        self.seq_type_event_hawkes_numpy = numpy.copy(
            self.seq_type_event_numpy[1:,:]
        )
        self.seq_sims_time_to_current_hawkes_numpy = numpy.zeros(
            (self.max_num_sims, self.max_len, self.size_batch), dtype=dtype
        )
        self.seq_sims_mask_to_current_hawkes_numpy = numpy.zeros(
            (self.max_num_sims, self.max_len, self.size_batch), dtype=dtype
        )
        #
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            seq = self.data[self.tag_batch][idx_data]
            num_sims_this_seq = numpy.int32(
                self.num_sims_start_to_end_numpy[idx_in_batch]
            )
            #
            temp_column = numpy.copy(
                self.sampled_times_numpy[:,idx_in_batch]
            )
            for idx_pos_prime, item_event_prime in enumerate(seq):
                t_i_prime = item_event_prime[
                    'time_since_start'
                ]
                indices_to_edit = (temp_column > t_i_prime)
                self.seq_sims_time_to_current_hawkes_numpy[
                    indices_to_edit, idx_pos_prime, idx_in_batch
                ] = (temp_column - t_i_prime)[indices_to_edit]
                self.seq_sims_mask_to_current_hawkes_numpy[
                    indices_to_edit, idx_pos_prime, idx_in_batch
                ] = numpy.float32(1.0)
        #
        #
    #
    #
    #

    def process_seq_hawkesinhib(
        self, multiple=numpy.int32(10), predict_first=True
    ):
        self.process_seq_hawkes(predict_first)
        '''
        then, prepare the samples tensors
        '''
        self.sample_times(multiple)

    #
    def process_seq_nan(self, predict_first):
        #print "getting batch ... "
        self.max_len = 0
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            len_seq = len(
                self.data[self.tag_batch][idx_data]
            )
            if self.max_len < len_seq:
                self.max_len = len_seq
        #
        #self.seq_time_to_end_numpy = numpy.zeros(
        #    (self.max_len, self.size_batch), dtype=dtype
        #)
        self.seq_time_to_current_numpy = numpy.zeros(
            (self.max_len, self.size_batch),
            dtype=dtype
        )
        self.seq_type_event_numpy = numpy.zeros(
            (self.max_len, self.size_batch),
            dtype=numpy.int32
        )
        self.seq_mask_numpy = numpy.zeros(
            (self.max_len-1, self.size_batch), dtype = dtype
        )
        #
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            seq = self.data[self.tag_batch][idx_data]
            #
            time_end = seq[-1]['time_since_start']
            #
            for idx_pos, item_event in enumerate(seq):
                #
                self.seq_type_event_numpy[
                    idx_pos, idx_in_batch
                ] = numpy.int32(item_event['type_event'] )
                #TODO: sometimes, we do not want to predict the first one, so we only predict starting from the second ...
                if idx_pos > 0:
                    self.seq_mask_numpy[
                        idx_pos-1, idx_in_batch
                    ] = numpy.float32(1.0)
                #
                #
                if self.partial_predict:
                    if item_event['type_event'] not in self.substream:
                        self.seq_mask_numpy[idx_pos-1, idx_in_batch] = numpy.float32(0.0)
                #
                #
                self.seq_time_to_current_numpy[
                    idx_pos, idx_in_batch
                ] = item_event['time_since_last_event']
                #
            #
        #
    #
    #
    def process_seq_neural_non_sample(self, predict_first):
        #print "getting batch ... "
        self.max_len = 0
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            #TODO: print for debug : floating point exp
            #print "seq is : ", self.data[self.tag_batch][idx_data]
            len_seq = len(
                self.data[self.tag_batch][idx_data]
            )
            if self.max_len < len_seq:
                self.max_len = len_seq
        #
        #self.seq_time_to_end_numpy = numpy.zeros(
        #    (self.max_len, self.size_batch), dtype=dtype
        #)
        #print "prepare array ... "
        if self.max_len < 1:
            self.max_len = numpy.int32(1)
        #
        self.seq_time_to_current_numpy = numpy.zeros(
            (self.max_len, self.size_batch),
            dtype=dtype
        ) #
        self.seq_type_event_numpy = numpy.zeros(
            (self.max_len+numpy.int32(1), self.size_batch),
            dtype=numpy.int32
        ) #
        self.seq_time_rep_numpy = numpy.zeros(
            (
                self.max_len+numpy.int32(1),
                self.size_batch, self.dim_time
            ), dtype = dtype
        ) #
        #
        self.seq_time_values_numpy = numpy.zeros(
            (
                self.max_len+numpy.int32(1),
                self.size_batch
            ), dtype = dtype
        )
        #
        self.time_since_start_to_end_numpy = numpy.zeros(
            (self.size_batch,), dtype=dtype
        ) #
        self.seq_mask_numpy = numpy.zeros(
            (self.max_len, self.size_batch), dtype = dtype
        ) #
        #
        self.num_events_start_to_end_numpy = numpy.zeros(
            (self.size_batch, ), dtype = dtype
        ) #
        #self.num_sims_start_to_end = numpy.zeros(
        #    (self.size_batch, ), dtype=dtype
        #)
        #
        #print "fill in the array"
        #print "one shape : ", self.seq_time_to_current_numpy.shape
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            seq = self.data[self.tag_batch][idx_data]
            #
            if len(seq) > 0:
                time_end = seq[-1]['time_since_start']
                #
                self.time_since_start_to_end_numpy[
                    idx_in_batch
                ] = time_end
                #
                self.num_events_start_to_end_numpy[
                    idx_in_batch
                ] = numpy.float32(len(seq) )
            else:
                time_end = numpy.float32(0.0)
                #
                self.time_since_start_to_end_numpy[
                    idx_in_batch
                ] = time_end
                #
                self.num_events_start_to_end_numpy[
                    idx_in_batch
                ] = numpy.float32(1)
            #self.num_sims_start_to_end[
            #    idx_in_batch
            #] = numpy.float32(
            #    multiple * self.num_events_start_to_end[idx_in_batch]
            #)
            self.seq_type_event_numpy[
                numpy.int32(0), idx_in_batch
            ] = numpy.int32(self.dim_process)
            #
            for idx_pos, item_event in enumerate(seq):
                #
                t_i = item_event['time_since_start']
                #
                self.seq_type_event_numpy[
                    idx_pos+numpy.int32(1), idx_in_batch
                ] = numpy.int32(item_event['type_event'] )
                #
                self.seq_mask_numpy[
                    idx_pos, idx_in_batch
                ] = numpy.float32(1.0)
                #
                #
                if self.partial_predict:
                    if item_event['type_event'] not in self.substream:
                        self.seq_mask_numpy[idx_pos, idx_in_batch] = numpy.float32(0.0)
                #
                #
                self.seq_time_to_current_numpy[
                    idx_pos, idx_in_batch
                ] = item_event['time_since_last_event']
                self.seq_time_rep_numpy[
                    idx_pos+numpy.int32(1), idx_in_batch, :
                ] = self.float32_to_bit(
                    item_event['time_since_last_event']
                )
                self.seq_time_values_numpy[
                    idx_pos+numpy.int32(1), idx_in_batch
                ] = item_event['time_since_last_event']
                #
                #
            #
        #
        #TODO: sometimes,
        #we do not want to predict the first one,
        #so we only predict starting from the second ...
        #example : all datasets from GaTech
        #if idx_pos > 0:
        if not predict_first:
            self.seq_mask_numpy[0,:] = numpy.float32(0.0)
        #


    def sample_times_neural(
        self, multiple=numpy.int32(10)
    ):
        self.num_sims_start_to_end_numpy = numpy.float32(
            multiple * self.num_events_start_to_end_numpy
        )
        self.max_num_sims = numpy.int32(
            numpy.max(
                self.num_sims_start_to_end_numpy
            )
        )
        self.seq_sims_time_to_current_numpy = numpy.zeros(
            (self.max_num_sims, self.size_batch),
            dtype = dtype
        )
        self.seq_sims_index_in_hidden_numpy = numpy.zeros(
            (self.max_num_sims, self.size_batch),
            dtype = numpy.int32
        )
        self.seq_sims_mask_numpy = numpy.zeros(
            (self.max_num_sims, self.size_batch),
            dtype = dtype
        )
        #
        '''
        this is used to compute intensity
        we generate numpy tensors for neural model
        then we use the same sampled times
        to generate numpy tensors for non-neural model
        '''
        self.sampled_times_numpy = numpy.zeros(
            (self.max_num_sims, self.size_batch),
            dtype=dtype
        )
        #
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            seq = self.data[self.tag_batch][idx_data]
            num_sims_this_seq = numpy.int32(
                self.num_sims_start_to_end_numpy[idx_in_batch]
            )
            #
            self.seq_sims_mask_numpy[
                :num_sims_this_seq, idx_in_batch
            ] = numpy.float32(1.0)
            #
            sampled_times = numpy.sort(
                numpy.random.uniform(
                    low = 0.0,
                    high = self.time_since_start_to_end_numpy[idx_in_batch],
                    size = (num_sims_this_seq,)
                )
            )
            temp_column = numpy.zeros(
                (self.max_num_sims,), dtype=dtype
            )
            temp_column[:num_sims_this_seq] = numpy.copy(
                sampled_times
            )
            #
            self.sampled_times_numpy[:,idx_in_batch] = numpy.copy(temp_column)
            #
            # for the sim times that < t_1
            # the time to current is sim times
            # for others
            # the time to current is s_j - t_rec
            if len(seq) > 0:
                t_1 = seq[0]['time_since_start']
            else:
                t_1 = numpy.float32(0.0)
            indices_to_edit = ( temp_column < t_1 )
            self.seq_sims_time_to_current_numpy[
                indices_to_edit, idx_in_batch
            ] = numpy.copy(temp_column[indices_to_edit] )
            '''
            T * size_batch * dim_model |-->
            (T * size_batch) * dim_model
            '''
            self.seq_sims_index_in_hidden_numpy[
                indices_to_edit, idx_in_batch
            ] = numpy.int32(
                idx_in_batch
            )
            #
            for idx_pos_prime, item_event_prime in enumerate(seq):
                '''
                fill in the :
                seq_sims_time_to_current_numpy
                seq_sims_index_in_hidden_numpy
                '''
                t_i_prime = item_event_prime[
                    'time_since_start'
                ]
                indices_to_edit = (temp_column > t_i_prime)
                self.seq_sims_time_to_current_numpy[
                    indices_to_edit, idx_in_batch
                ] = numpy.copy(
                    (temp_column - t_i_prime)[indices_to_edit]
                )
                '''
                the assignment here is tricky
                needs to be double checked
                '''
                self.seq_sims_index_in_hidden_numpy[
                    indices_to_edit, idx_in_batch
                ] = numpy.int32(
                    (idx_pos_prime + numpy.int32(1) ) * self.size_batch + idx_in_batch
                )
                '''
                we need to +1 since 0 -- self.size_batch is taken by the first couple of samples
                in details, say if idx_pos_prime = 0
                t_i_prime = 0.5
                you want to use the representation of the LSTM after feeding 0-th event
                but there is a BOS event, so you need the 1-th representation
                so we need to +1
                '''
                '''
                the + 1 won't cause index out of range
                since there is no temp_column > t_I
                due to how data is sampled
                '''
                '''
                need to edit here :
                for each t_i, find the sim times s_j that
                s_j > t_i and s_j < t_i+1
                then assign the part to the right index
                when assigning index, we need to consider
                reshape and flatten
                '''
                #
                #
            #
        #
    #
    #

    def process_seq_neural(
        self, multiple=numpy.int32(10), predict_first=True
    ):
        self.process_seq_neural_non_sample(predict_first)
        self.sample_times_neural(multiple)

    def process_data(
        self, tag_batch, idx_batch_current=0,
        tag_model = 'hawkes',
        multiple = numpy.int32(10),
        predict_first = True
    ):
        #
        self.tag_batch = tag_batch
        self.list_idx_data = [
            idx for idx in self.list_idx[self.tag_batch][
                idx_batch_current * self.size_batch : (idx_batch_current + 1) * self.size_batch
            ]
        ]
        assert(
            tag_model == 'neural' or tag_model == 'hawkes' or tag_model == 'hawkesinhib' or tag_model == 'nanmodel'
        )
        #
        if tag_model == 'hawkes':
            self.process_seq_hawkes(predict_first)
        elif tag_model == 'hawkesinhib':
            self.process_seq_hawkesinhib(
                multiple, predict_first
            )
        elif tag_model == 'neural':
            self.process_seq_neural(
                multiple, predict_first
            )
        elif tag_model == 'nanmodel':
            self.process_seq_nan(predict_first)
        else:
            print "wrong model tag"
    #
    #
    def process_data_lambda(
        self, tag_batch, idx_batch_current=0,
        multiple = numpy.int32(10),
        predict_first = True
    ):
        #
        self.tag_batch = tag_batch
        self.list_idx_data = [
            idx for idx in self.list_idx[self.tag_batch][
                idx_batch_current * self.size_batch : (idx_batch_current + 1) * self.size_batch
            ]
        ]
        #
        self.process_seq_neural(
            multiple, predict_first
        )
        self.get_tensors_hawkes_given_sampled_times()
    #
    #
    #

    # for logs

    def creat_log(self, log_dict):
        '''
        log dict keys : log_file, compile_time, things need to be tracked
        '''
        print "creating training log file ... "
        current_log_file = os.path.abspath(
            log_dict['log_file']
        )
        with open(current_log_file, 'w') as f:
            f.write('This the training log file ... \n')
            f.write('It tracks some statistics in the training process ... \n')
            #
            f.write('Model specs are listed below : \n')
            for the_key in log_dict['args']:
                f.write(
                    the_key+' : '+str(log_dict['args'][the_key])
                )
                f.write('\n')
            #
            f.write('Before training, the compilation time is '+str(log_dict['compile_time'])+' sec ... \n')
            f.write('Things that need to be tracked : \n')
            for the_key in log_dict['tracked']:
                f.write(the_key+' ')
            f.write('\n\n')
        #

    def continue_log(self, log_dict):
        print "continue tracking log ... "
        current_log_file = os.path.abspath(
            log_dict['log_file']
        )
        with open(current_log_file, 'a') as f:
            for the_key in log_dict['tracked']:
                f.write(the_key+' is '+str(log_dict['tracked'][the_key])+' \n')
            #
            if log_dict['what_to_track'] == 'loss':
                if log_dict['max_dev_log_likelihood'] < log_dict['tracked']['dev_log_likelihood']:
                    f.write('This is a new best model ! \n')
                    log_dict['max_dev_log_likelihood'] = log_dict['tracked']['dev_log_likelihood']
                    #
                    # update the tracked_best
                    for the_key in log_dict['tracked']:
                        log_dict['tracked_best'][
                            the_key
                        ] = log_dict['tracked'][the_key]
                    #
            elif log_dict['what_to_track'] == 'rmse':
                if log_dict['tracked']['dev_rmse'] < log_dict['min_dev_rmse']:
                    f.write('This is a new best model ! \n')
                    log_dict['min_dev_rmse'] = log_dict['tracked']['dev_rmse']
                    #
                    # update the tracked_best
                    for the_key in log_dict['tracked']:
                        log_dict['tracked_best'][
                            the_key
                        ] = log_dict['tracked'][the_key]
            elif log_dict['what_to_track'] == 'rate':
                if log_dict['tracked']['dev_error_rate'] < log_dict['min_dev_error_rate']:
                    f.write('This is a new best model ! \n')
                    log_dict['min_dev_error_rate'] = log_dict['tracked']['dev_error_rate']
                    #
                    # update the tracked_best
                    for the_key in log_dict['tracked']:
                        log_dict['tracked_best'][
                            the_key
                        ] = log_dict['tracked'][the_key]
            else:
                print "what to track ?"
            #
            f.write('\n')

    def finish_log(self, log_dict):
        print "finish tracking log ... "
        current_log_file = os.path.abspath(
            log_dict['log_file']
        )
        with open(current_log_file, 'a') as f:
            f.write('The best model info is shown below : \n')
            for the_key in log_dict['tracked_best']:
                f.write(
                    the_key+' is '+str(log_dict['tracked_best'][the_key])+' \n'
                )
                #
            f.write('\n')

    def track_log(self, log_dict):
        #print "recording training log ... "
        '''
        log dict keys : log_file, compile_time, things need to be tracked
        '''
        assert(log_dict['mode']=='create' or log_dict['mode']=='continue')
        if log_dict['mode'] == 'create':
            self.creat_log(log_dict)
        else:
            self.continue_log(log_dict)
