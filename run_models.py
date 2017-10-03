# -*- coding: utf-8 -*-
"""
Created on Mar 18th 10:58:37 2016

run models, including training and validating

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
import modules.utils as utils
import modules.models as models
import modules.optimizers as optimizers
import modules.controllers as controllers
import modules.data_processers as data_processers
import modules.organizers as organizers

dtype=theano.config.floatX


#
# Hawkes process
def train_hawkes_ctsm(input_train):
    '''
    this function is called to train hawkes ctsm model
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_train['seed_random']
    )
    #
    #command_mkdir = 'mkdir -p ' + os.path.abspath(
    #    input_train['save_file_path']
    #)
    #os.system(command_mkdir)
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': input_train['save_file_path'],
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        'min_dev_error_rate': 1e6,
        'min_dev_rmse': 1e6,
        #
        'what_to_track': input_train['what_to_track'],
        #
        'args': input_train['args'],
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        #
        'tracked_best': {},
        #
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_train['path_rawdata'],
            'size_batch': input_train['size_batch'],
            'ratio_train': input_train['ratio_train'],
            'to_read': [
                'train', 'dev'
            ],
            'partial_predict': input_train['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_train['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'model': input_train['model'],
        'dim_process': data_process.dim_process,
        'coef_l2': input_train['coef_l2'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train'],
        'learn_rate': input_train['learn_rate'],
        'predict_lambda': input_train['predict_lambda']
    }

    control = controllers.ControlHawkesCTSM(
        model_settings
    )

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        total_log_likelihood, total_log_likelihood_time, total_log_likelihood_type = 0.0, 0.0, 0.0
        total_num_of_events = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                'train', step_train,
                tag_model = 'hawkes',
                predict_first = input_train['predict_first']
            )
            #
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_learn(
                data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_mask_to_current_numpy
            )
            #
            #
            log_dict['iteration'] += 1
            #
            total_log_likelihood += log_likelihood_numpy
            total_log_likelihood_time += log_likelihood_time_numpy
            total_log_likelihood_type += log_likelihood_type_numpy
            total_num_of_events += num_of_events_numpy
            #
            #
            log_dict['tracked']['train_log_likelihood'] = round(
                total_log_likelihood / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_time'] = round(
                total_log_likelihood_time / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type'] = round(
                total_log_likelihood_type / total_num_of_events, 4
            )
            train_end = time.time()
            log_dict['tracked']['train_time'] = round(
                (train_end - train_start)*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            #
            #print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: go through the dev data and calculate the dev metrics
                print "Now we start validating after batches ", log_dict['track_period']
                dev_start = time.time()
                #
                #TODO: get the dev loss values
                total_log_likelihood_dev = 0.0
                total_log_likelihood_time_dev = 0.0
                total_log_likelihood_type_dev = 0.0
                total_num_of_events_dev = 0.0
                for step_dev in range(data_process.max_nums['dev']):
                    #
                    data_process.process_data(
                        'dev', step_dev,
                        tag_model = 'hawkes',
                        predict_first = input_train['predict_first']
                    )
                    #
                    #print "validating ... "
                    log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                        data_process.seq_time_to_end_numpy,
                        data_process.seq_time_to_current_numpy,
                        data_process.seq_type_event_numpy,
                        data_process.time_since_start_to_end_numpy,
                        data_process.seq_mask_numpy,
                        data_process.seq_mask_to_current_numpy
                    )
                    #
                    total_log_likelihood_dev += log_likelihood_numpy
                    total_log_likelihood_time_dev += log_likelihood_time_numpy
                    total_log_likelihood_type_dev += log_likelihood_type_numpy
                    total_num_of_events_dev += num_of_events_numpy
                    #
                    if step_dev % 10 == 9:
                        print "in dev, the step is out of ", step_dev, data_process.max_nums['dev']
                    #
                    #print "in dev, the step is out of ", step_dev, data_process.max_nums['dev']
                #
                #
                log_dict['tracked']['dev_log_likelihood'] = round(
                    total_log_likelihood_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_time'] = round(
                    total_log_likelihood_time_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_type'] = round(
                    total_log_likelihood_type_dev / total_num_of_events_dev, 4
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round( dev_end - dev_start, 0 )
                #
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                #
                if log_dict['tracked']['dev_log_likelihood'] > log_dict['max_dev_log_likelihood']:
                    #
                    #name_file = 'model_'+str(log_dict['tracked']['track_cnt'])+'.pkl'
                    name_file = 'model.pkl'
                    save_file = os.path.abspath(
                        log_dict['save_file_path']
                    ) + '/'+name_file
                    #
                    control.save_model(save_file)
                #
                data_process.track_log(log_dict)
            ########
    data_process.finish_log(log_dict)
    print "finish training"
    #
#
#
def test_hawkes_ctsm_and_save(input_test):
    '''
    this function is called to test hawkes ctsm model
    and save
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_test['seed_random']
    )
    #
    log_dict = {
        'log_file': None,
        'save_file_path': None,
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        #
        'args': input_test['args'],
        #
        'iteration': 0,
        'track_period': None,
        'max_epoch': input_test['max_epoch'],
        'size_batch': input_test['size_batch'],
        # this size batch has to be 1 !!!
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_test['path_rawdata'],
            'size_batch': input_test['size_batch'],
            'ratio_train': numpy.float32(0.0),
            'to_read': [input_test['tag_split']],
            'partial_predict': input_test['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_test['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    # in testing, regularizer is not used,
    # but we still pass it in, to make sure the model is built
    model_settings = {
        'model': input_test['model'],
        'dim_process': data_process.dim_process,
        'coef_l2': input_test['coef_l2'],
        'size_batch': input_test['size_batch'],
        #'size_batch': numpy.int32(1),
        'optimizer': input_test['optimizer'],
        'path_pre_train': input_test['path_pre_train'],
        'predict_lambda': input_test['predict_lambda']
    }

    control = controllers.ControlHawkesCTSM(
        model_settings
    )

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    tag_split = input_test['tag_split']

    for epi in range(log_dict['max_epoch']):
        #
        list_log_likelihood_seq = []
        list_log_likelihood_type = []
        list_log_likelihood_time = []
        list_num_of_events = []
        #
        dev_start = time.time()
        print "validating for ", tag_split
        #
        #TODO: get the dev loss values
        total_log_likelihood_dev = 0.0
        total_log_likelihood_time_dev = 0.0
        total_log_likelihood_type_dev = 0.0
        total_num_of_events_dev = 0.0
        for step_dev in range(data_process.max_nums[tag_split]):
            #
            data_process.process_data(
                tag_split, step_dev,
                tag_model = 'hawkes',
                predict_first = input_test['predict_first']
            )
            #
            #print "validating for ", tag_split
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_mask_to_current_numpy
            )
            #
            list_log_likelihood_seq.append(
                log_likelihood_numpy
            )
            list_log_likelihood_type.append(
                log_likelihood_type_numpy
            )
            list_log_likelihood_time.append(
                log_likelihood_time_numpy
            )
            list_num_of_events.append(
                num_of_events_numpy
            )
            #
            total_log_likelihood_dev += log_likelihood_numpy
            total_log_likelihood_time_dev += log_likelihood_time_numpy
            total_log_likelihood_type_dev += log_likelihood_type_numpy
            total_num_of_events_dev += num_of_events_numpy
            #
            if step_dev % 100 == 99:
                print "in validation, the step is out of ", step_dev, data_process.max_nums[tag_split]
            #
        log_likelihood_final = round(
            total_log_likelihood_dev / total_num_of_events_dev, 4
        )
        log_likelihood_time_final = round(
            total_log_likelihood_time_dev / total_num_of_events_dev, 4
        )
        log_likelihood_type_final = round(
            total_log_likelihood_type_dev / total_num_of_events_dev, 4
        )
        #
        dev_end = time.time()
        #
        # start saving stuff !!!
        dict_results = {
            'log_likelihood': {
                'seq': log_likelihood_final,
                'type': log_likelihood_type_final,
                'time': log_likelihood_time_final
            },
            'model': input_test['path_pre_train'],
            'data': input_test['path_rawdata'],
            'tag_split': tag_split,
            'lists': {
                'log_likelihood_seq': list_log_likelihood_seq,
                'log_likelihood_type': list_log_likelihood_type,
                'log_likelihood_time': list_log_likelihood_time,
                'num_of_events': list_num_of_events
            }
        }
        #
        #TODO: add more info about this model
        if input_test['path_logs'] != None:
            log_org = organizers.LogOrg(
                {
                    'path_tracks': os.path.abspath(
                        input_test['path_logs']
                    )
                }
            )
            log_org.read_logs()
            dict_log_org = log_org.get_one_log()
            dict_results['log_info'] = dict_log_org
        else:
            dict_results['log_info'] = None
        #
        with open(input_test['file_to_save_results'], 'wb') as f:
            pickle.dump(dict_results, f)
        #
        #
        print "the model is : ", input_test['path_pre_train']
        print "the dataset is : ", input_test['path_rawdata']
        print "the tag split is : ", tag_split
        print "the log-likelihood of seq is : ", log_likelihood_final
        print "the log-likelihood of type is : ", log_likelihood_type_final
        print "the log-likelihood of time is : ", log_likelihood_time_final
        print "the three values in the order of seq / type / time : is " + str(log_likelihood_final) + ' / ' + str(log_likelihood_type_final) + ' / ' + str(log_likelihood_time_final)
        #
        #
    print "finish testing"
#
#
def test_intensity_and_save(input_test):
    '''
    this function is called to test hawkes
    evaluation on intensity prediciton
    and save results
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_test['seed_random']
    )
    #
    log_dict = {
        'log_file': None,
        'save_file_path': None,
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        #
        'args': input_test['args'],
        #
        'iteration': 0,
        'track_period': None,
        'max_epoch': input_test['max_epoch'],
        'size_batch': input_test['size_batch'],
        # this size batch has to be 1 !!!
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_test['path_rawdata'],
            'size_batch': input_test['size_batch'],
            'ratio_train': numpy.float32(0.0),
            'to_read': [input_test['tag_split']],
            'partial_predict': input_test['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_test['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    # in testing, regularizer is not used,
    # but we still pass it in, to make sure the model is built
    model_settings = {
        'model': input_test['model'],
        'dim_process': data_process.dim_process,
        'coef_l2': input_test['coef_l2'],
        'size_batch': input_test['size_batch'],
        #'size_batch': numpy.int32(1),
        'optimizer': input_test['optimizer'],
        'path_pre_train': input_test['path_pre_train'],
        'predict_lambda': input_test['predict_lambda'],
        'loss_type': 'loglikehood'
    }
    #
    model_settings_gold = {
        'model': input_test['model_gold'],
        'dim_process': data_process.dim_process,
        'coef_l2': input_test['coef_l2'],
        'size_batch': input_test['size_batch'],
        'optimizer': input_test['optimizer'],
        'path_pre_train': input_test['path_gold'],
        'predict_lambda': input_test['predict_lambda'],
        'loss_type': 'loglikehood'
    }
    #
    #TODO: make controller given model
    if input_test['model'] == 'hawkes':
        control = controllers.ControlHawkesCTSM(
            model_settings
        )
    elif input_test['model'] == 'neuralreduce':
        control = controllers.ControlNeuralHawkesAdaptiveBaseCTSM_time(
            model_settings
        )
    elif input_test['model'] == 'conttime':
        control = controllers.ControlNeuralHawkesAdaptiveBaseCTSM_time(
            model_settings
        )
    else:
        print "wrong model, it is gonna crash"
    #
    if input_test['model_gold'] == 'hawkes':
        control_gold = controllers.ControlHawkesCTSM(
            model_settings_gold
        )
    elif input_test['model_gold'] == 'neuralreduce':
        control_gold = controllers.ControlNeuralHawkesAdaptiveBaseCTSM_time(
            model_settings_gold
        )
    elif input_test['model_gold'] == 'conttime':
        control_gold = controllers.ControlNeuralHawkesAdaptiveBaseCTSM_time(
            model_settings_gold
        )
    else:
        print "wrong model it is gonna crash"
    #
    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    tag_split = input_test['tag_split']

    for epi in range(log_dict['max_epoch']):
        #
        list_lambda_samples = []
        list_lambda_samples_gold = []
        list_num_of_samples = []
        #
        dev_start = time.time()
        print "validating for ", tag_split
        #
        #TODO: get the dev loss values
        for step_dev in range(data_process.max_nums[tag_split]):
            #
            data_process.process_data_lambda(
                tag_split, step_dev,
                predict_first = input_test['predict_first']
            )
            #
            #print "validating for ", tag_split
            if input_test['model'] == 'hawkes':
                lambda_samples_numpy, num_of_samples_numpy = control.model_dev_lambda(
                    data_process.seq_type_event_hawkes_numpy,
                    data_process.seq_sims_time_to_current_hawkes_numpy,
                    data_process.seq_sims_mask_numpy,
                    data_process.seq_sims_mask_to_current_hawkes_numpy
                )
            else:
                '''
                by default it is neural
                can add others like hawkesinhib
                '''
                lambda_samples_numpy, num_of_samples_numpy = control.model_dev_lambda(
                    data_process.seq_type_event_numpy,
                    data_process.seq_time_values_numpy,
                    data_process.seq_sims_time_to_current_numpy,
                    data_process.seq_sims_index_in_hidden_numpy,
                    data_process.seq_sims_mask_numpy
                )
            #
            if input_test['model_gold']=='hawkes':
                lambda_samples_gold_numpy, num_of_samples_gold_numpy = control_gold.model_dev_lambda(
                    data_process.seq_type_event_hawkes_numpy,
                    data_process.seq_sims_time_to_current_hawkes_numpy,
                    data_process.seq_sims_mask_numpy,
                    data_process.seq_sims_mask_to_current_hawkes_numpy
                )
            else:
                '''
                by default it is neural
                can add others like hawkesinhib
                '''
                lambda_samples_gold_numpy, num_of_samples_gold_numpy = control_gold.model_dev_lambda(
                    data_process.seq_type_event_numpy,
                    data_process.seq_time_values_numpy,
                    data_process.seq_sims_time_to_current_numpy,
                    data_process.seq_sims_index_in_hidden_numpy,
                    data_process.seq_sims_mask_numpy
                )
            #
            list_lambda_samples.append(
                numpy.copy(lambda_samples_numpy)
            )
            list_lambda_samples_gold.append(
                numpy.copy(
                    lambda_samples_gold_numpy
                )
            )
            list_num_of_samples.append(
                numpy.copy(num_of_samples_numpy)
            )
            #
            if step_dev % 100 == 99:
                print "in validation, the step is out of ", step_dev, data_process.max_nums[tag_split]
            #
        #
        dev_end = time.time()
        #
        # start saving stuff !!!
        dict_results = {
            'model': input_test['model'],
            'model_gold': input_test['model_gold'],
            'path_pre_train': input_test['path_pre_train'],
            'path_gold': input_test['path_gold'],
            'data': input_test['path_rawdata'],
            'tag_split': tag_split,
            'lists': {
                'lambda_samples': list_lambda_samples,
                'lambda_samples_gold': list_lambda_samples_gold,
                'num_of_samples': list_num_of_samples
            }
        }
        #
        #TODO: add more info about this model
        #if input_test['path_logs'] != None:
        #    log_org = organizers.LogOrg(
        #        {
        #            'path_tracks': os.path.abspath(
        #                input_test['path_logs']
        #            )
        #        }
        #    )
        #    log_org.read_logs()
        #    dict_log_org = log_org.get_one_log()
        #    dict_results['log_info'] = dict_log_org
        #else:
        #    dict_results['log_info'] = None
        #
        with open(input_test['file_to_save_results'], 'wb') as f:
            pickle.dump(dict_results, f)
        #
        #
        print "the model is : ", input_test['path_pre_train']
        print "the gold model is : ", input_test['path_gold']
        print "the dataset is : ", input_test['path_rawdata']
        print "the tag split is : ", tag_split
        #
    print "finish testing"
    #
#
# Hawkes process with inhibition
def train_hawkesinhib_ctsm(input_train):
    '''
    this function is called to train hawkes ctsm model with inhibition
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_train['seed_random']
    )
    #
    #command_mkdir = 'mkdir -p ' + os.path.abspath(
    #    input_train['save_file_path']
    #)
    #os.system(command_mkdir)
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': input_train['save_file_path'],
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        'min_dev_error_rate': 1e6,
        'min_dev_rmse': 1e6,
        #
        'what_to_track': input_train['what_to_track'],
        #
        'args': input_train['args'],
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        #
        'tracked_best': {},
        #
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_train['path_rawdata'],
            'size_batch': input_train['size_batch'],
            'ratio_train': input_train['ratio_train'],
            'to_read': [
                'train', 'dev'
            ],
            'partial_predict': input_train['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_train['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'model': input_train['model'],
        'dim_process': data_process.dim_process,
        'coef_l2': input_train['coef_l2'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train'],
        'predict_lambda': input_train['predict_lambda']
    }

    control = controllers.ControlHawkesInhibCTSM(
        model_settings
    )

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        total_log_likelihood, total_log_likelihood_time, total_log_likelihood_type = 0.0, 0.0, 0.0
        total_num_of_events = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                tag_batch = 'train',
                idx_batch_current = step_train,
                tag_model = 'hawkesinhib',
                multiple = numpy.int32(
                    input_train['multiple_sample_for_train']
                ),
                predict_first = input_train['predict_first']
            )
            #
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_learn(
                #data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_mask_to_current_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_mask_to_current_numpy,
                data_process.seq_sims_mask_numpy
            )
            #
            #
            log_dict['iteration'] += 1
            #
            total_log_likelihood += log_likelihood_numpy
            total_log_likelihood_time += log_likelihood_time_numpy
            total_log_likelihood_type += log_likelihood_type_numpy
            total_num_of_events += num_of_events_numpy
            #
            #
            log_dict['tracked']['train_log_likelihood'] = round(
                total_log_likelihood / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_time'] = round(
                total_log_likelihood_time / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type'] = round(
                total_log_likelihood_type / total_num_of_events, 4
            )
            train_end = time.time()
            log_dict['tracked']['train_time'] = round(
                (train_end - train_start)*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: go through the dev data and calculate the dev metrics
                print "Now we start validating after batches ", log_dict['track_period']
                dev_start = time.time()
                #
                #TODO: get the dev loss values
                total_log_likelihood_dev = 0.0
                total_log_likelihood_time_dev = 0.0
                total_log_likelihood_type_dev = 0.0
                total_num_of_events_dev = 0.0
                for step_dev in range(data_process.max_nums['dev']):
                    #
                    data_process.process_data(
                        tag_batch = 'dev',
                        idx_batch_current = step_dev,
                        tag_model = 'hawkesinhib',
                        multiple = numpy.int32(
                            input_train[
                                'multiple_sample_for_dev'
                            ]
                        ),
                        predict_first = input_train['predict_first']
                    )
                    #
                    #print "validating ... "
                    log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                        #data_process.seq_time_to_end_numpy,
                        data_process.seq_time_to_current_numpy,
                        data_process.seq_type_event_numpy,
                        data_process.time_since_start_to_end_numpy,
                        data_process.num_sims_start_to_end_numpy,
                        data_process.seq_mask_numpy,
                        data_process.seq_mask_to_current_numpy,
                        data_process.seq_sims_time_to_current_numpy,
                        data_process.seq_sims_mask_to_current_numpy,
                        data_process.seq_sims_mask_numpy
                    )
                    #
                    total_log_likelihood_dev += log_likelihood_numpy
                    total_log_likelihood_time_dev += log_likelihood_time_numpy
                    total_log_likelihood_type_dev += log_likelihood_type_numpy
                    total_num_of_events_dev += num_of_events_numpy
                    #
                    if step_dev % 10 == 9:
                        print "in dev, the step is out of ", step_dev, data_process.max_nums['dev']
                #
                #
                log_dict['tracked']['dev_log_likelihood'] = round(
                    total_log_likelihood_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_time'] = round(
                    total_log_likelihood_time_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_type'] = round(
                    total_log_likelihood_type_dev / total_num_of_events_dev, 4
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round( dev_end - dev_start, 0 )
                #
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                #
                if log_dict['tracked']['dev_log_likelihood'] > log_dict['max_dev_log_likelihood']:
                    #
                    #name_file = 'model_'+str(log_dict['tracked']['track_cnt'])+'.pkl'
                    name_file = 'model.pkl'
                    save_file = os.path.abspath(
                        log_dict['save_file_path']
                    ) + '/'+name_file
                    #
                    control.save_model(save_file)
                #
                data_process.track_log(log_dict)
            ########
    data_process.finish_log(log_dict)
    print "finish training"
    #
#
#
def test_hawkesinhib_ctsm_and_save(input_test):
    '''
    this function is called to test hawkes ctsm model with inhibition
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_test['seed_random']
    )
    #
    log_dict = {
        'log_file': None,
        'save_file_path': None,
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        #
        'args': input_test['args'],
        #
        'iteration': 0,
        'track_period': None,
        'max_epoch': input_test['max_epoch'],
        'size_batch': input_test['size_batch'],
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_test['path_rawdata'],
            'size_batch': input_test['size_batch'],
            'ratio_train': numpy.float32(0.0),
            'to_read': [input_test['tag_split']],
            'partial_predict': input_test['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_test['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'model': input_test['model'],
        'dim_process': data_process.dim_process,
        'coef_l2': input_test['coef_l2'],
        'size_batch': input_test['size_batch'],
        'optimizer': input_test['optimizer'],
        'path_pre_train': input_test['path_pre_train'],
        'predict_lambda': input_test['predict_lambda']
    }

    control = controllers.ControlHawkesInhibCTSM(
        model_settings
    )

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    tag_split = input_test['tag_split']

    for epi in range(log_dict['max_epoch']):
        #
        list_log_likelihood_seq = []
        list_log_likelihood_type = []
        list_log_likelihood_time = []
        list_num_of_events = []
        #
        dev_start = time.time()
        #
        print "validating for ", tag_split
        #
        total_log_likelihood_dev = 0.0
        total_log_likelihood_time_dev = 0.0
        total_log_likelihood_type_dev = 0.0
        total_num_of_events_dev = 0.0
        #
        for step_dev in range(data_process.max_nums[tag_split] ):
            #
            data_process.process_data(
                tag_batch = tag_split,
                idx_batch_current = step_dev,
                tag_model = 'hawkesinhib',
                multiple = numpy.int32(
                    input_test['multiple_sample_for_dev']
                ),
                predict_first = input_test['predict_first']
            )
            #
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                #data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_mask_to_current_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_mask_to_current_numpy,
                data_process.seq_sims_mask_numpy
            )
            #
            #
            list_log_likelihood_seq.append(
                log_likelihood_numpy
            )
            list_log_likelihood_type.append(
                log_likelihood_type_numpy
            )
            list_log_likelihood_time.append(
                log_likelihood_time_numpy
            )
            list_num_of_events.append(
                num_of_events_numpy
            )
            #
            #
            total_log_likelihood_dev += log_likelihood_numpy
            total_log_likelihood_time_dev += log_likelihood_time_numpy
            total_log_likelihood_type_dev += log_likelihood_type_numpy
            total_num_of_events_dev += num_of_events_numpy
            #
            if step_dev % 100 == 99:
                print "in validation, the step is out of ", step_dev, data_process.max_nums[tag_split]
            #
        log_likelihood_final = round(
            total_log_likelihood_dev / total_num_of_events_dev, 4
        )
        log_likelihood_time_final = round(
            total_log_likelihood_time_dev / total_num_of_events_dev, 4
        )
        log_likelihood_type_final = round(
            total_log_likelihood_type_dev / total_num_of_events_dev, 4
        )
        #
        dev_end = time.time()
        #
        # start saving stuff !!!
        dict_results = {
            'log_likelihood': {
                'seq': log_likelihood_final,
                'type': log_likelihood_type_final,
                'time': log_likelihood_time_final
            },
            'model': input_test['path_pre_train'],
            'data': input_test['path_rawdata'],
            'tag_split': tag_split,
            'lists': {
                'log_likelihood_seq': list_log_likelihood_seq,
                'log_likelihood_type': list_log_likelihood_type,
                'log_likelihood_time': list_log_likelihood_time,
                'num_of_events': list_num_of_events
            }
        }
        #
        #TODO: add more info about this model
        if input_test['path_logs'] != None:
            log_org = organizers.LogOrg(
                {
                    'path_tracks': os.path.abspath(
                        input_test['path_logs']
                    )
                }
            )
            log_org.read_logs()
            dict_log_org = log_org.get_one_log()
            dict_results['log_info'] = dict_log_org
        else:
            dict_results['log_info'] = None
        #
        #
        with open(input_test['file_to_save_results'], 'wb') as f:
            pickle.dump(dict_results, f)
        #
        #
        print "the model is : ", input_test['path_pre_train']
        print "the dataset is : ", input_test['path_rawdata']
        print "the tag split is : ", tag_split
        print "the log-likelihood of seq is : ", log_likelihood_final
        print "the log-likelihood of type is : ", log_likelihood_type_final
        print "the log-likelihood of time is : ", log_likelihood_time_final
        #
        print "the three values in the order of seq / type / time : is " + str(log_likelihood_final) + ' / ' + str(log_likelihood_type_final) + ' / ' + str(log_likelihood_time_final)
        #
    print "finish testing"
#
#
#
def train_generalized_neural_hawkes_ctsm_time(
    input_train, tag_neural_type = 'general'
):
    '''
    this function is called to train
    generalized neural hawkes ctsm
    tag can be : general, adaptive, simple
    though simple is deprecated
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_train['seed_random']
    )
    #
    assert(
        tag_neural_type == 'general' or tag_neural_type == 'adaptive' or tag_neural_type == 'simple'
    )
    #
    assert(
        input_train['loss_type'] == 'loglikehood' or input_train['loss_type'] == 'prediction'
    )
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': input_train['save_file_path'],
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        'min_dev_error_rate': 1e6,
        'min_dev_rmse': 1e6,
        #
        'what_to_track': input_train['what_to_track'],
        #
        'args': input_train['args'],
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        #
        'tracked_best': {},
        #
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_log_likelihood_type_predict': None,
            'dev_log_likelihood_type_predict': None,
            'train_rmse': None,
            'dev_rmse': None,
            'train_error_rate': None,
            'dev_error_rate': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_train['path_rawdata'],
            'size_batch': input_train['size_batch'],
            'ratio_train': input_train['ratio_train'],
            'to_read': [
                'train', 'dev'
            ],
            'partial_predict': input_train['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_train['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    #
    print "get time quantiles ... "
    data_process.get_time_quantiles()
    #

    model_settings = {
        'model': input_train['model'],
        'loss_type': input_train['loss_type'],
        'dim_process': data_process.dim_process,
        #
        'dim_time': data_process.dim_time,
        'dim_model': input_train['dim_model'],
        #
        'coef_l2': input_train['coef_l2'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train'],
        'threshold_time': numpy.copy(
            data_process.time_quantiles
        ),
        'learn_rate': input_train['learn_rate'],
        'predict_lambda': input_train['predict_lambda']
    }
    #
    if tag_neural_type == 'general':
        control = controllers.ControlGeneralizedNeuralHawkesCTSM_time(
            model_settings
        )
    elif tag_neural_type == 'adaptive':
        control = controllers.ControlNeuralHawkesAdaptiveBaseCTSM_time(
            model_settings
        )
    elif tag_neural_type == 'simple':
        print "simple version of neural hawkes with time encoder NOT implemented ... "
    else:
        print "not implemented"
    #
    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        total_log_likelihood = 0.0
        total_log_likelihood_time = 0.0
        total_log_likelihood_type = 0.0
        total_log_likelihood_type_predict = 0.0
        total_num_of_events = 0.0
        total_num_of_errors = 0.0
        total_square_errors = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                tag_batch = 'train',
                idx_batch_current = step_train,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_train['multiple_sample_for_train']
                ),
                predict_first = input_train['predict_first']
            )
            #
            time_diffs_numpy = numpy.float32(
                numpy.array(
                    sorted(
                        numpy.random.exponential(
                            scale=1.0,
                            size=(100,)
                        )
                    )
                )
            )
            #
            log_likelihood_numpy = 0.0
            log_likelihood_time_numpy = 0.0
            log_likelihood_type_numpy = 0.0
            log_likelihood_type_predict_numpy = 0.0
            num_of_events_numpy = 0.0
            num_of_errors_numpy = 0.0
            square_errors_numpy = 0.0
            #
            if input_train['loss_type'] == 'loglikehood':
                log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_learn(
                    data_process.seq_time_to_current_numpy,
                    data_process.seq_type_event_numpy,
                    data_process.seq_time_values_numpy,
                    data_process.time_since_start_to_end_numpy,
                    data_process.num_sims_start_to_end_numpy,
                    data_process.seq_mask_numpy,
                    data_process.seq_sims_time_to_current_numpy,
                    data_process.seq_sims_index_in_hidden_numpy,
                    data_process.seq_sims_mask_numpy
                )
            else:
                log_likelihood_type_predict_numpy, num_of_errors_numpy, square_errors_numpy, num_of_events_numpy = control.model_learn(
                    data_process.seq_type_event_numpy,
                    data_process.seq_time_values_numpy,
                    data_process.seq_mask_numpy,
                    time_diffs_numpy
                )
                #print "gradient absoluate value : ", grad_numpy
            #
            #
            log_dict['iteration'] += 1
            #
            total_log_likelihood += log_likelihood_numpy
            total_log_likelihood_time += log_likelihood_time_numpy
            total_log_likelihood_type += log_likelihood_type_numpy
            total_log_likelihood_type_predict += log_likelihood_type_predict_numpy
            total_num_of_events += num_of_events_numpy
            total_num_of_errors += num_of_errors_numpy
            total_square_errors += square_errors_numpy
            #
            #
            log_dict['tracked']['train_log_likelihood'] = round(
                total_log_likelihood / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_time'] = round(
                total_log_likelihood_time / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type'] = round(
                total_log_likelihood_type / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type_predict'] = round(
                total_log_likelihood_type_predict / total_num_of_events, 4
            )
            log_dict['tracked']['train_rmse'] = round(
                numpy.sqrt(
                    total_square_errors / total_num_of_events
                ), 8
            )
            log_dict['tracked']['train_error_rate'] = round(
                total_num_of_errors / total_num_of_events, 4
            )
            train_end = time.time()
            #
            log_dict['tracked']['train_time'] = round(
                (train_end - train_start)*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: go through the dev data and calculate the dev metrics
                print "Now we start validating after batches ", log_dict['track_period']
                dev_start = time.time()
                #
                #TODO: get the dev loss values
                total_log_likelihood_dev = 0.0
                total_log_likelihood_time_dev = 0.0
                total_log_likelihood_type_dev = 0.0
                total_log_likelihood_type_predict_dev = 0.0
                total_num_of_events_dev = 0.0
                total_num_of_errors_dev = 0.0
                total_square_errors_dev = 0.0
                #
                for step_dev in range(data_process.max_nums['dev']):
                    #
                    #
                    data_process.process_data(
                        tag_batch = 'dev',
                        idx_batch_current = step_dev,
                        tag_model = 'neural',
                        multiple = numpy.int32(
                            input_train[
                                'multiple_sample_for_dev'
                            ]
                        ),
                        predict_first = input_train['predict_first']
                    )
                    #
                    time_diffs_numpy = numpy.float32(
                        numpy.array(
                            sorted(
                                numpy.random.exponential(
                                    scale=1.0,
                                    size=(100,)
                                )
                            )
                        )
                    )
                    #
                    #
                    log_likelihood_numpy = 0.0
                    log_likelihood_time_numpy = 0.0
                    log_likelihood_type_numpy = 0.0
                    log_likelihood_type_predict_numpy = 0.0
                    num_of_events_numpy = 0.0
                    num_of_errors_numpy = 0.0
                    square_errors_numpy = 0.0
                    #
                    #
                    if input_train['loss_type'] == 'loglikehood':
                        log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                            data_process.seq_time_to_current_numpy,
                            data_process.seq_type_event_numpy,
                            data_process.seq_time_values_numpy,
                            data_process.time_since_start_to_end_numpy,
                            data_process.num_sims_start_to_end_numpy,
                            data_process.seq_mask_numpy,
                            data_process.seq_sims_time_to_current_numpy,
                            data_process.seq_sims_index_in_hidden_numpy,
                            data_process.seq_sims_mask_numpy
                        )
                    else:
                        log_likelihood_type_predict_numpy, num_of_errors_numpy, square_errors_numpy, num_of_events_numpy = control.model_dev(
                            data_process.seq_type_event_numpy,
                            data_process.seq_time_values_numpy,
                            data_process.seq_mask_numpy,
                            time_diffs_numpy
                        )
                        #print "gradient absoluate value : ", grad_numpy
                        #
                    #
                    total_log_likelihood_dev += log_likelihood_numpy
                    total_log_likelihood_time_dev += log_likelihood_time_numpy
                    total_log_likelihood_type_dev += log_likelihood_type_numpy
                    total_log_likelihood_type_predict_dev += log_likelihood_type_predict_numpy
                    total_num_of_events_dev += num_of_events_numpy
                    total_num_of_errors_dev += num_of_errors_numpy
                    total_square_errors_dev += square_errors_numpy
                    #
                    if step_dev % 10 == 9:
                        print "in dev, the step is out of ", step_dev, data_process.max_nums['dev']
                #
                #
                log_dict['tracked']['dev_log_likelihood'] = round(
                    total_log_likelihood_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_time'] = round(
                    total_log_likelihood_time_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_type'] = round(
                    total_log_likelihood_type_dev / total_num_of_events_dev, 4
                )
                #
                log_dict['tracked']['dev_log_likelihood_type_predict'] = round(
                    total_log_likelihood_type_predict_dev / total_num_of_events_dev, 4
                )
                #
                log_dict['tracked']['dev_error_rate'] = round(
                    total_num_of_errors_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_rmse'] = round(
                    numpy.sqrt(
                        total_square_errors_dev / total_num_of_events_dev
                    ), 8
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round(
                    dev_end - dev_start, 0
                )
                #
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                if log_dict['what_to_track'] == 'loss':
                    if log_dict['tracked']['dev_log_likelihood'] > log_dict['max_dev_log_likelihood']:
                        #
                        #name_file = 'model_'+str(log_dict['tracked']['track_cnt'])+'.pkl'
                        name_file = 'model.pkl'
                        save_file = os.path.abspath(
                            log_dict['save_file_path']
                        ) + '/'+name_file
                        #
                        control.save_model(save_file)
                elif log_dict['what_to_track'] == 'rmse':
                    if log_dict['tracked']['dev_rmse'] < log_dict['min_dev_rmse']:
                        name_file = 'model.pkl'
                        save_file = os.path.abspath(
                            log_dict['save_file_path']
                        ) + '/'+name_file
                        #
                        control.save_model(save_file)
                elif log_dict['what_to_track'] == 'rate':
                    if log_dict['tracked']['dev_error_rate'] < log_dict['min_dev_error_rate']:
                        name_file = 'model.pkl'
                        save_file = os.path.abspath(
                            log_dict['save_file_path']
                        ) + '/'+name_file
                        #
                        control.save_model(save_file)
                else:
                    print "what tracker ? "
                #
                data_process.track_log(log_dict)
            ########
    data_process.finish_log(log_dict)
    print "finish training"
    #
    #
#
#
#
#
def train_generalized_neural_hawkes_ctsm_time_DevIncludedSetting(
    input_train, tag_neural_type = 'general'
):
    '''
    this function is only called
    when compared with prev work on MIMIC, SO and FINANCIAL datasets
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_train['seed_random']
    )
    #
    assert(
        tag_neural_type == 'general' or tag_neural_type == 'adaptive' or tag_neural_type == 'simple'
    )
    #
    assert(
        input_train['loss_type'] == 'loglikehood' or input_train['loss_type'] == 'prediction'
    )
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': input_train['save_file_path'],
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        'min_dev_error_rate': 1e6,
        'min_dev_rmse': 1e6,
        #
        'what_to_track': input_train['what_to_track'],
        #
        'args': input_train['args'],
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        #
        'tracked_best': {},
        #
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_log_likelihood_type_predict': None,
            'dev_log_likelihood_type_predict': None,
            'train_rmse': None,
            'dev_rmse': None,
            'train_error_rate': None,
            'dev_error_rate': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_train['path_rawdata'],
            'size_batch': input_train['size_batch'],
            'ratio_train': input_train['ratio_train'],
            'to_read': [
                'train', 'dev', 'test'
            ],
            'partial_predict': input_train['partial_predict']
        }
    )
    data_process.re_init()
    #
    prune_stream = numpy.int32(
        input_train['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    #
    print "get time quantiles ... "
    data_process.get_time_quantiles()
    #

    model_settings = {
        'model': input_train['model'],
        'loss_type': input_train['loss_type'],
        'dim_process': data_process.dim_process,
        #
        'dim_time': data_process.dim_time,
        'dim_model': input_train['dim_model'],
        #
        'coef_l2': input_train['coef_l2'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train'],
        'threshold_time': numpy.copy(
            data_process.time_quantiles
        ),
        'learn_rate': input_train['learn_rate'],
        'predict_lambda': input_train['predict_lambda']
    }
    #
    if tag_neural_type == 'general':
        control = controllers.ControlGeneralizedNeuralHawkesCTSM_time(
            model_settings
        )
    elif tag_neural_type == 'adaptive':
        control = controllers.ControlNeuralHawkesAdaptiveBaseCTSM_time(
            model_settings
        )
    elif tag_neural_type == 'simple':
        print "simple version of neural hawkes with time encoder NOT implemented ... "
    else:
        print "not implemented"
    #
    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        total_log_likelihood = 0.0
        total_log_likelihood_time = 0.0
        total_log_likelihood_type = 0.0
        total_log_likelihood_type_predict = 0.0
        total_num_of_events = 0.0
        total_num_of_errors = 0.0
        total_square_errors = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                tag_batch = 'train',
                idx_batch_current = step_train,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_train['multiple_sample_for_train']
                ),
                predict_first = input_train['predict_first']
            )
            #
            time_diffs_numpy = numpy.float32(
                numpy.array(
                    sorted(
                        numpy.random.exponential(
                            scale=1.0,
                            size=(100,)
                        )
                    )
                )
            )
            #
            log_likelihood_numpy = 0.0
            log_likelihood_time_numpy = 0.0
            log_likelihood_type_numpy = 0.0
            log_likelihood_type_predict_numpy = 0.0
            num_of_events_numpy = 0.0
            num_of_errors_numpy = 0.0
            square_errors_numpy = 0.0
            #
            if input_train['loss_type'] == 'loglikehood':
                log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_learn(
                    data_process.seq_time_to_current_numpy,
                    data_process.seq_type_event_numpy,
                    data_process.seq_time_values_numpy,
                    data_process.time_since_start_to_end_numpy,
                    data_process.num_sims_start_to_end_numpy,
                    data_process.seq_mask_numpy,
                    data_process.seq_sims_time_to_current_numpy,
                    data_process.seq_sims_index_in_hidden_numpy,
                    data_process.seq_sims_mask_numpy
                )
            else:
                log_likelihood_type_predict_numpy, num_of_errors_numpy, square_errors_numpy, num_of_events_numpy = control.model_learn(
                    data_process.seq_type_event_numpy,
                    data_process.seq_time_values_numpy,
                    data_process.seq_mask_numpy,
                    time_diffs_numpy
                )
                #print "gradient absoluate value : ", grad_numpy
            #
            #
            log_dict['iteration'] += 1
            #
            total_log_likelihood += log_likelihood_numpy
            total_log_likelihood_time += log_likelihood_time_numpy
            total_log_likelihood_type += log_likelihood_type_numpy
            total_log_likelihood_type_predict += log_likelihood_type_predict_numpy
            total_num_of_events += num_of_events_numpy
            total_num_of_errors += num_of_errors_numpy
            total_square_errors += square_errors_numpy
            #
            #
            log_dict['tracked']['train_log_likelihood'] = round(
                total_log_likelihood / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_time'] = round(
                total_log_likelihood_time / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type'] = round(
                total_log_likelihood_type / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type_predict'] = round(
                total_log_likelihood_type_predict / total_num_of_events, 4
            )
            log_dict['tracked']['train_rmse'] = round(
                numpy.sqrt(
                    total_square_errors / total_num_of_events
                ), 8
            )
            log_dict['tracked']['train_error_rate'] = round(
                total_num_of_errors / total_num_of_events, 4
            )
            train_end = time.time()
            #
            log_dict['tracked']['train_time'] = round(
                (train_end - train_start)*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: track the training ...
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                #
                print "Now we start testing after training one epoch ... "
                dev_start = time.time()
                #
                #TODO: get the dev loss values
                total_log_likelihood_dev = 0.0
                total_log_likelihood_time_dev = 0.0
                total_log_likelihood_type_dev = 0.0
                total_log_likelihood_type_predict_dev = 0.0
                total_num_of_events_dev = 0.0
                total_num_of_errors_dev = 0.0
                total_square_errors_dev = 0.0
                #
                for step_dev in range(data_process.max_nums['dev']):
                    #
                    #
                    data_process.process_data(
                        tag_batch = 'dev',
                        idx_batch_current = step_dev,
                        tag_model = 'neural',
                        multiple = numpy.int32(
                            input_train[
                                'multiple_sample_for_dev'
                            ]
                        ),
                        predict_first = input_train['predict_first']
                    )
                    #
                    time_diffs_numpy = numpy.float32(
                        numpy.array(
                            sorted(
                                numpy.random.exponential(
                                    scale=1.0,
                                    size=(100,)
                                )
                            )
                        )
                    )
                    #
                    #
                    log_likelihood_numpy = 0.0
                    log_likelihood_time_numpy = 0.0
                    log_likelihood_type_numpy = 0.0
                    log_likelihood_type_predict_numpy = 0.0
                    num_of_events_numpy = 0.0
                    num_of_errors_numpy = 0.0
                    square_errors_numpy = 0.0
                    #
                    #
                    if input_train['loss_type'] == 'loglikehood':
                        log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                            data_process.seq_time_to_current_numpy,
                            data_process.seq_type_event_numpy,
                            data_process.seq_time_values_numpy,
                            data_process.time_since_start_to_end_numpy,
                            data_process.num_sims_start_to_end_numpy,
                            data_process.seq_mask_numpy,
                            data_process.seq_sims_time_to_current_numpy,
                            data_process.seq_sims_index_in_hidden_numpy,
                            data_process.seq_sims_mask_numpy
                        )
                    else:
                        log_likelihood_type_predict_numpy, num_of_errors_numpy, square_errors_numpy, num_of_events_numpy = control.model_dev(
                            data_process.seq_type_event_numpy,
                            data_process.seq_time_values_numpy,
                            data_process.seq_mask_numpy,
                            time_diffs_numpy
                        )
                        #print "gradient absoluate value : ", grad_numpy
                        #
                    #
                    total_log_likelihood_dev += log_likelihood_numpy
                    total_log_likelihood_time_dev += log_likelihood_time_numpy
                    total_log_likelihood_type_dev += log_likelihood_type_numpy
                    total_log_likelihood_type_predict_dev += log_likelihood_type_predict_numpy
                    total_num_of_events_dev += num_of_events_numpy
                    total_num_of_errors_dev += num_of_errors_numpy
                    total_square_errors_dev += square_errors_numpy
                    #
                    if step_dev % 10 == 9:
                        print "in dev, the step is out of ", step_dev, data_process.max_nums['dev']
                #
                #
                name_file = 'model'+str(
                    log_dict['tracked']['track_cnt']
                )+'.pkl'
                save_file = os.path.abspath(
                    log_dict['save_file_path']
                ) + '/'+name_file
                #
                control.save_model(save_file)
                #
                log_dict['tracked']['dev_log_likelihood'] = round(
                    total_log_likelihood_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_time'] = round(
                    total_log_likelihood_time_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_type'] = round(
                    total_log_likelihood_type_dev / total_num_of_events_dev, 4
                )
                #
                log_dict['tracked']['dev_log_likelihood_type_predict'] = round(
                    total_log_likelihood_type_predict_dev / total_num_of_events_dev, 4
                )
                #
                log_dict['tracked']['dev_error_rate'] = round(
                    total_num_of_errors_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_rmse'] = round(
                    numpy.sqrt(
                        total_square_errors_dev / total_num_of_events_dev
                    ), 8
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round(
                    dev_end - dev_start, 0
                )
                #
                data_process.track_log(log_dict)
                #
            ########
    data_process.finish_log(log_dict)
    print "finish training"
    #
    #
#
#
#
#
def test_generalized_neural_hawkes_ctsm_and_save_time(
    input_test, tag_neural_type = 'general'
):
    '''
    this function is called to test
    generalized neural hawkes ctsm with time encoder
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_test['seed_random']
    )
    #
    assert(
        tag_neural_type == 'neural' or tag_neural_type == 'general' or tag_neural_type == 'adaptive' or tag_neural_type == 'simple'
    )
    #
    log_dict = {
        'log_file': None,
        'save_file_path': None,
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        #
        'args': input_test['args'],
        #
        'iteration': 0,
        'track_period': None,
        'max_epoch': input_test['max_epoch'],
        'size_batch': input_test['size_batch'],
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "
    #
    #
    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_test['path_rawdata'],
            'size_batch': input_test['size_batch'],
            'ratio_train': numpy.float32(0.0),
            'to_read': [input_test['tag_split']],
            'partial_predict': input_test['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_test['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    #
    #print "get time quantiles ... "
    #data_process.get_time_quantiles()
    #

    model_settings = {
        'model': input_test['model'],
        'loss_type': input_test['loss_type'],
        'dim_process': data_process.dim_process,
        #
        #'dim_time': data_process.dim_time,
        #'dim_model': input_test['dim_model'],
        #
        'coef_l2': input_test['coef_l2'],
        'size_batch': input_test['size_batch'],
        'optimizer': input_test['optimizer'],
        'path_pre_train': input_test['path_pre_train'],
        'predict_lambda': input_test['predict_lambda']
    }

    #
    if tag_neural_type == 'neural':
        control = controllers.ControlNeuralHawkesCTSM_time(
            model_settings
        )
    elif tag_neural_type == 'general':
        control = controllers.ControlGeneralizedNeuralHawkesCTSM_time(
            model_settings
        )
    elif tag_neural_type == 'adaptive':
        control = controllers.ControlNeuralHawkesAdaptiveBaseCTSM_time(
            model_settings
        )
    elif tag_neural_type == 'simple':
        print "simple version of neural hawkes with time encoder NOT implemented ... "
    else:
        print "not implemented"
    #

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    tag_split = input_test['tag_split']

    for epi in range(log_dict['max_epoch']):
        #
        list_log_likelihood_seq = []
        list_log_likelihood_type = []
        list_log_likelihood_time = []
        list_num_of_events = []
        #
        dev_start = time.time()
        #
        print "validating for ", tag_split
        #
        total_log_likelihood_dev = 0.0
        total_log_likelihood_time_dev = 0.0
        total_log_likelihood_type_dev = 0.0
        total_num_of_events_dev = 0.0
        #
        for step_dev in range(data_process.max_nums[tag_split] ):
            #
            #TODO; print for debug : floating point exp
            #print "step : ", step_dev
            #
            data_process.process_data(
                tag_batch = tag_split,
                idx_batch_current = step_dev,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_test['multiple_sample_for_dev']
                ),
                predict_first = input_test['predict_first']
            )
            #
            #print "validating ... "
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                #data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                #data_process.seq_time_rep_numpy,
                data_process.seq_time_values_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_index_in_hidden_numpy,
                data_process.seq_sims_mask_numpy
            )
            #
            list_log_likelihood_seq.append(
                log_likelihood_numpy
            )
            list_log_likelihood_type.append(
                log_likelihood_type_numpy
            )
            list_log_likelihood_time.append(
                log_likelihood_time_numpy
            )
            list_num_of_events.append(
                num_of_events_numpy
            )
            #
            total_log_likelihood_dev += log_likelihood_numpy
            total_log_likelihood_time_dev += log_likelihood_time_numpy
            total_log_likelihood_type_dev += log_likelihood_type_numpy
            total_num_of_events_dev += num_of_events_numpy
            #
            if step_dev % 100 == 99:
                print "in validation, the step is out of ", step_dev, data_process.max_nums[tag_split]
            #
        log_likelihood_final = round(
            total_log_likelihood_dev / total_num_of_events_dev, 4
        )
        log_likelihood_time_final = round(
            total_log_likelihood_time_dev / total_num_of_events_dev, 4
        )
        log_likelihood_type_final = round(
            total_log_likelihood_type_dev / total_num_of_events_dev, 4
        )
        #
        dev_end = time.time()
        #
        # start saving stuff !!!
        dict_results = {
            'log_likelihood': {
                'seq': log_likelihood_final,
                'type': log_likelihood_type_final,
                'time': log_likelihood_time_final
            },
            'model': input_test['path_pre_train'],
            'data': input_test['path_rawdata'],
            'tag_split': tag_split,
            'lists': {
                'log_likelihood_seq': list_log_likelihood_seq,
                'log_likelihood_type': list_log_likelihood_type,
                'log_likelihood_time': list_log_likelihood_time,
                'num_of_events': list_num_of_events
            }
        }
        #
        #TODO: add more info about this model
        if input_test['path_logs'] != None:
            log_org = organizers.LogOrg(
                {
                    'path_tracks': os.path.abspath(
                        input_test['path_logs']
                    )
                }
            )
            log_org.read_logs()
            dict_log_org = log_org.get_one_log()
            dict_results['log_info'] = dict_log_org
        #
        #
        with open(input_test['file_to_save_results'], 'wb') as f:
            pickle.dump(dict_results, f)
        #
        #
        print "the model is : ", input_test['path_pre_train']
        print "the dataset is : ", input_test['path_rawdata']
        print "the tag split is : ", tag_split
        print "the log-likelihood of seq is : ", log_likelihood_final
        print "the log-likelihood of type is : ", log_likelihood_type_final
        print "the log-likelihood of time is : ", log_likelihood_time_final

        #
        print "the three values in the order of seq / type / time : is " + str(log_likelihood_final) + ' / ' + str(log_likelihood_type_final) + ' / ' + str(log_likelihood_time_final)
        #
    print "finish testing and saving "
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
# deprecated functions
# TODO: some working but not useful anymore
#
#
#
# Neural Hawkes process -- effect estimated by LSTM
def train_neural_hawkes_ctsm(input_train):
    '''
    this function is called to train neural hawkes ctsm
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_train['seed_random']
    )
    #
    #command_mkdir = 'mkdir -p ' + os.path.abspath(
    #    input_train['save_file_path']
    #)
    #os.system(command_mkdir)
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': input_train['save_file_path'],
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        'min_dev_error_rate': 1e6,
        'min_dev_rmse': 1e6,
        #
        'what_to_track': input_train['what_to_track'],
        #
        'args': input_train['args'],
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        #
        'tracked_best': {},
        #
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_train['path_rawdata'],
            'size_batch': input_train['size_batch'],
            'ratio_train': input_train['ratio_train'],
            'to_read': [
                'train', 'dev'
            ],
            'partial_predict': input_train['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_train['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'model': input_train['model'],
        'dim_process': data_process.dim_process,
        #
        'dim_time': data_process.dim_time,
        'dim_model': input_train['dim_model'],
        #
        'coef_l2': input_train['coef_l2'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train'],
        'predict_lambda': input_train['predict_lambda']
    }
    #
    control = controllers.ControlNeuralHawkesCTSM(
        model_settings
    )
    #
    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        total_log_likelihood = 0.0
        total_log_likelihood_time = 0.0
        total_log_likelihood_type = 0.0
        total_num_of_events = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                tag_batch = 'train',
                idx_batch_current = step_train,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_train['multiple_sample_for_train']
                ),
                predict_first = input_train['predict_first']
            )
            #
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_learn(
                #data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                data_process.seq_time_rep_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_index_in_hidden_numpy,
                data_process.seq_sims_mask_numpy
            )
            #
            #
            log_dict['iteration'] += 1
            #
            total_log_likelihood += log_likelihood_numpy
            total_log_likelihood_time += log_likelihood_time_numpy
            total_log_likelihood_type += log_likelihood_type_numpy
            total_num_of_events += num_of_events_numpy
            #
            #
            log_dict['tracked']['train_log_likelihood'] = round(
                total_log_likelihood / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_time'] = round(
                total_log_likelihood_time / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type'] = round(
                total_log_likelihood_type / total_num_of_events, 4
            )
            train_end = time.time()
            #
            log_dict['tracked']['train_time'] = round(
                (train_end - train_start)*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: go through the dev data and calculate the dev metrics
                print "Now we start validating after batches ", log_dict['track_period']
                dev_start = time.time()
                #
                #TODO: get the dev loss values
                total_log_likelihood_dev = 0.0
                total_log_likelihood_time_dev = 0.0
                total_log_likelihood_type_dev = 0.0
                total_num_of_events_dev = 0.0
                for step_dev in range(data_process.max_nums['dev']):
                    #
                    data_process.process_data(
                        tag_batch = 'dev',
                        idx_batch_current = step_dev,
                        tag_model = 'neural',
                        multiple = numpy.int32(
                            input_train[
                                'multiple_sample_for_dev'
                            ]
                        ),
                        predict_first = input_train['predict_first']
                    )
                    #
                    #print "validating ... "
                    log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                        #data_process.seq_time_to_end_numpy,
                        data_process.seq_time_to_current_numpy,
                        data_process.seq_type_event_numpy,
                        data_process.seq_time_rep_numpy,
                        data_process.time_since_start_to_end_numpy,
                        data_process.num_sims_start_to_end_numpy,
                        data_process.seq_mask_numpy,
                        data_process.seq_sims_time_to_current_numpy,
                        data_process.seq_sims_index_in_hidden_numpy,
                        data_process.seq_sims_mask_numpy
                    )
                    #
                    total_log_likelihood_dev += log_likelihood_numpy
                    total_log_likelihood_time_dev += log_likelihood_time_numpy
                    total_log_likelihood_type_dev += log_likelihood_type_numpy
                    total_num_of_events_dev += num_of_events_numpy
                    #
                    if step_dev % 10 == 9:
                        print "in dev, the step is out of ", step_dev, data_process.max_nums['dev']
                #
                #
                log_dict['tracked']['dev_log_likelihood'] = round(
                    total_log_likelihood_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_time'] = round(
                    total_log_likelihood_time_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_type'] = round(
                    total_log_likelihood_type_dev / total_num_of_events_dev, 4
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round( dev_end - dev_start, 0 )
                #
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                #
                if log_dict['tracked']['dev_log_likelihood'] > log_dict['max_dev_log_likelihood']:
                    #
                    #name_file = 'model_'+str(log_dict['tracked']['track_cnt'])+'.pkl'
                    name_file = 'model.pkl'
                    save_file = os.path.abspath(
                        log_dict['save_file_path']
                    ) + '/'+name_file
                    #
                    control.save_model(save_file)
                #
                data_process.track_log(log_dict)
            ########
    data_process.finish_log(log_dict)
    print "finish training"
    #
    #
#
#
def test_neural_hawkes_ctsm_and_save(input_test):
    '''
    this function is called to test neural hawkes ctsm
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_test['seed_random']
    )
    #
    log_dict = {
        'log_file': None,
        'save_file_path': None,
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        #
        'args': input_test['args'],
        #
        'iteration': 0,
        'track_period': None,
        'max_epoch': input_test['max_epoch'],
        'size_batch': input_test['size_batch'],
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_test['path_rawdata'],
            'size_batch': input_test['size_batch'],
            'ratio_train': numpy.float32(0.0),
            'to_read': [input_test['tag_split']],
            'partial_predict': input_test['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_test['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'model': input_test['model'],
        'dim_process': data_process.dim_process,
        #
        #'dim_time': data_process.dim_time,
        #'dim_model': input_test['dim_model'],
        #
        'coef_l2': input_test['coef_l2'],
        'size_batch': input_test['size_batch'],
        'optimizer': input_test['optimizer'],
        'path_pre_train': input_test['path_pre_train'],
        'predict_lambda': input_test['predict_lambda']
    }

    control = controllers.ControlNeuralHawkesCTSM(
        model_settings
    )

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    tag_split = input_test['tag_split']

    for epi in range(log_dict['max_epoch']):
        #
        list_log_likelihood_seq = []
        list_log_likelihood_type = []
        list_log_likelihood_time = []
        list_num_of_events = []
        #
        dev_start = time.time()
        #
        print "validating for ", tag_split
        #
        total_log_likelihood_dev = 0.0
        total_log_likelihood_time_dev = 0.0
        total_log_likelihood_type_dev = 0.0
        total_num_of_events_dev = 0.0
        #
        for step_dev in range(data_process.max_nums[tag_split] ):
            #
            data_process.process_data(
                tag_batch = tag_split,
                idx_batch_current = step_dev,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_test['multiple_sample_for_dev']
                ),
                predict_first = input_test['predict_first']
            )
            #
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                #data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                data_process.seq_time_rep_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_index_in_hidden_numpy,
                data_process.seq_sims_mask_numpy
            )
            #
            list_log_likelihood_seq.append(
                log_likelihood_numpy
            )
            list_log_likelihood_type.append(
                log_likelihood_type_numpy
            )
            list_log_likelihood_time.append(
                log_likelihood_time_numpy
            )
            list_num_of_events.append(
                num_of_events_numpy
            )
            #
            total_log_likelihood_dev += log_likelihood_numpy
            total_log_likelihood_time_dev += log_likelihood_time_numpy
            total_log_likelihood_type_dev += log_likelihood_type_numpy
            total_num_of_events_dev += num_of_events_numpy
            #
            if step_dev % 100 == 99:
                print "in validation, the step is out of ", step_dev, data_process.max_nums[tag_split]
            #
        log_likelihood_final = round(
            total_log_likelihood_dev / total_num_of_events_dev, 4
        )
        log_likelihood_time_final = round(
            total_log_likelihood_time_dev / total_num_of_events_dev, 4
        )
        log_likelihood_type_final = round(
            total_log_likelihood_type_dev / total_num_of_events_dev, 4
        )
        #
        dev_end = time.time()
        #
        # start saving stuff !!!
        dict_results = {
            'log_likelihood': {
                'seq': log_likelihood_final,
                'type': log_likelihood_type_final,
                'time': log_likelihood_time_final
            },
            'model': input_test['path_pre_train'],
            'data': input_test['path_rawdata'],
            'tag_split': tag_split,
            'lists': {
                'log_likelihood_seq': list_log_likelihood_seq,
                'log_likelihood_type': list_log_likelihood_type,
                'log_likelihood_time': list_log_likelihood_time,
                'num_of_events': list_num_of_events
            }
        }
        #
        #TODO: add more info about this model
        log_org = organizers.LogOrg(
            {
                'path_tracks': os.path.abspath(
                    input_test['path_logs']
                )
            }
        )
        log_org.read_logs()
        dict_log_org = log_org.get_one_log()
        dict_results['log_info'] = dict_log_org
        #
        #
        with open(input_test['file_to_save_results'], 'wb') as f:
            pickle.dump(dict_results, f)
        #
        #
        print "the model is : ", input_test['path_pre_train']
        print "the dataset is : ", input_test['path_rawdata']
        print "the tag split is : ", tag_split
        print "the log-likelihood of seq is : ", log_likelihood_final
        print "the log-likelihood of type is : ", log_likelihood_type_final
        print "the log-likelihood of time is : ", log_likelihood_time_final
        #
        print "the three values in the order of seq / type / time : is " + str(log_likelihood_final) + ' / ' + str(log_likelihood_type_final) + ' / ' + str(log_likelihood_time_final)
        #
    print "finish testing"
#
#
#
# Generalized Neural Hawkes process --
# effect and decay estimated by LSTM
#
def train_generalized_neural_hawkes_ctsm(
    input_train, tag_neural_type = 'general'
):
    '''
    this function is called to train
    generalized neural hawkes ctsm
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_train['seed_random']
    )
    #
    assert(
        tag_neural_type == 'general' or tag_neural_type == 'adaptive' or tag_neural_type == 'simple'
    )
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': input_train['save_file_path'],
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        'min_dev_error_rate': 1e6,
        'min_dev_rmse': 1e6,
        #
        'what_to_track': input_train['what_to_track'],
        #
        'args': input_train['args'],
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        #
        'tracked_best': {},
        #
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_train['path_rawdata'],
            'size_batch': input_train['size_batch'],
            'ratio_train': input_train['ratio_train'],
            'to_read': [
                'train', 'dev'
            ],
            'partial_predict': input_train['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_train['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'model': input_train['model'],
        'dim_process': data_process.dim_process,
        #
        'dim_time': data_process.dim_time,
        'dim_model': input_train['dim_model'],
        #
        'coef_l2': input_train['coef_l2'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train'],
        'predict_lambda': input_train['predict_lambda']
    }
    #
    if tag_neural_type == 'general':
        control = controllers.ControlGeneralizedNeuralHawkesCTSM(
            model_settings
        )
    elif tag_neural_type == 'adaptive':
        control = controllers.ControlNeuralHawkesAdaptiveBaseCTSM(
            model_settings
        )
    elif tag_neural_type == 'simple':
        control = controllers.ControlNeuralHawkesSimpleCTSM(
            model_settings
        )
    else:
        print "not implemented"
    #
    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        total_log_likelihood = 0.0
        total_log_likelihood_time = 0.0
        total_log_likelihood_type = 0.0
        total_num_of_events = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                tag_batch = 'train',
                idx_batch_current = step_train,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_train['multiple_sample_for_train']
                ),
                predict_first = input_train['predict_first']
            )
            #
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_learn(
                #data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                data_process.seq_time_rep_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_index_in_hidden_numpy,
                data_process.seq_sims_mask_numpy
            )
            #
            #
            log_dict['iteration'] += 1
            #
            total_log_likelihood += log_likelihood_numpy
            total_log_likelihood_time += log_likelihood_time_numpy
            total_log_likelihood_type += log_likelihood_type_numpy
            total_num_of_events += num_of_events_numpy
            #
            #
            log_dict['tracked']['train_log_likelihood'] = round(
                total_log_likelihood / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_time'] = round(
                total_log_likelihood_time / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type'] = round(
                total_log_likelihood_type / total_num_of_events, 4
            )
            train_end = time.time()
            #
            log_dict['tracked']['train_time'] = round(
                (train_end - train_start)*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: go through the dev data and calculate the dev metrics
                print "Now we start validating after batches ", log_dict['track_period']
                dev_start = time.time()
                #
                #TODO: get the dev loss values
                total_log_likelihood_dev = 0.0
                total_log_likelihood_time_dev = 0.0
                total_log_likelihood_type_dev = 0.0
                total_num_of_events_dev = 0.0
                for step_dev in range(data_process.max_nums['dev']):
                    #
                    data_process.process_data(
                        tag_batch = 'dev',
                        idx_batch_current = step_dev,
                        tag_model = 'neural',
                        multiple = numpy.int32(
                            input_train[
                                'multiple_sample_for_dev'
                            ]
                        ),
                        predict_first = input_train['predict_first']
                    )
                    #
                    #print "validating ... "
                    log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                        #data_process.seq_time_to_end_numpy,
                        data_process.seq_time_to_current_numpy,
                        data_process.seq_type_event_numpy,
                        data_process.seq_time_rep_numpy,
                        data_process.time_since_start_to_end_numpy,
                        data_process.num_sims_start_to_end_numpy,
                        data_process.seq_mask_numpy,
                        data_process.seq_sims_time_to_current_numpy,
                        data_process.seq_sims_index_in_hidden_numpy,
                        data_process.seq_sims_mask_numpy
                    )
                    #
                    total_log_likelihood_dev += log_likelihood_numpy
                    total_log_likelihood_time_dev += log_likelihood_time_numpy
                    total_log_likelihood_type_dev += log_likelihood_type_numpy
                    total_num_of_events_dev += num_of_events_numpy
                    #
                    if step_dev % 10 == 9:
                        print "in dev, the step is out of ", step_dev, data_process.max_nums['dev']
                #
                #
                log_dict['tracked']['dev_log_likelihood'] = round(
                    total_log_likelihood_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_time'] = round(
                    total_log_likelihood_time_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_type'] = round(
                    total_log_likelihood_type_dev / total_num_of_events_dev, 4
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round( dev_end - dev_start, 0 )
                #
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                #
                if log_dict['tracked']['dev_log_likelihood'] > log_dict['max_dev_log_likelihood']:
                    #
                    #name_file = 'model_'+str(log_dict['tracked']['track_cnt'])+'.pkl'
                    name_file = 'model.pkl'
                    save_file = os.path.abspath(
                        log_dict['save_file_path']
                    ) + '/'+name_file
                    #
                    control.save_model(save_file)
                #
                data_process.track_log(log_dict)
            ########
    data_process.finish_log(log_dict)
    print "finish training"
    #
    #
#
#
def test_generalized_neural_hawkes_ctsm_and_save(
    input_test, tag_neural_type = 'general'
):
    '''
    this function is called to test
    generalized neural hawkes ctsm
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_test['seed_random']
    )
    #
    assert(
        tag_neural_type == 'general' or tag_neural_type == 'adaptive' or tag_neural_type == 'simple'
    )
    #
    log_dict = {
        'log_file': None,
        'save_file_path': None,
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        #
        'args': input_test['args'],
        #
        'iteration': 0,
        'track_period': None,
        'max_epoch': input_test['max_epoch'],
        'size_batch': input_test['size_batch'],
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_test['path_rawdata'],
            'size_batch': input_test['size_batch'],
            'ratio_train': numpy.float32(0.0),
            'to_read': [input_test['tag_split']],
            'partial_predict': input_test['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_test['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'model': input_test['model'],
        'dim_process': data_process.dim_process,
        #
        #'dim_time': data_process.dim_time,
        #'dim_model': input_test['dim_model'],
        #
        'coef_l2': input_test['coef_l2'],
        'size_batch': input_test['size_batch'],
        'optimizer': input_test['optimizer'],
        'path_pre_train': input_test['path_pre_train'],
        'predict_lambda': input_test['predict_lambda']
    }

    #
    if tag_neural_type == 'general':
        control = controllers.ControlGeneralizedNeuralHawkesCTSM(
            model_settings
        )
    elif tag_neural_type == 'adaptive':
        control = controllers.ControlNeuralHawkesAdaptiveBaseCTSM(
            model_settings
        )
    elif tag_neural_type == 'simple':
        control = controllers.ControlNeuralHawkesSimpleCTSM(
            model_settings
        )
    else:
        print "not implemented"
    #

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    tag_split = input_test['tag_split']

    for epi in range(log_dict['max_epoch']):
        #
        list_log_likelihood_seq = []
        list_log_likelihood_type = []
        list_log_likelihood_time = []
        list_num_of_events = []
        #
        dev_start = time.time()
        #
        print "validating for ", tag_split
        #
        total_log_likelihood_dev = 0.0
        total_log_likelihood_time_dev = 0.0
        total_log_likelihood_type_dev = 0.0
        total_num_of_events_dev = 0.0
        #
        for step_dev in range(data_process.max_nums[tag_split] ):
            #
            data_process.process_data(
                tag_batch = tag_split,
                idx_batch_current = step_dev,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_test['multiple_sample_for_dev']
                ),
                predict_first = input_test['predict_first']
            )
            #
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                #data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                data_process.seq_time_rep_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_index_in_hidden_numpy,
                data_process.seq_sims_mask_numpy
            )
            #
            list_log_likelihood_seq.append(
                log_likelihood_numpy
            )
            list_log_likelihood_type.append(
                log_likelihood_type_numpy
            )
            list_log_likelihood_time.append(
                log_likelihood_time_numpy
            )
            list_num_of_events.append(
                num_of_events_numpy
            )
            #
            total_log_likelihood_dev += log_likelihood_numpy
            total_log_likelihood_time_dev += log_likelihood_time_numpy
            total_log_likelihood_type_dev += log_likelihood_type_numpy
            total_num_of_events_dev += num_of_events_numpy
            #
            if step_dev % 100 == 99:
                print "in validation, the step is out of ", step_dev, data_process.max_nums[tag_split]
            #
        log_likelihood_final = round(
            total_log_likelihood_dev / total_num_of_events_dev, 4
        )
        log_likelihood_time_final = round(
            total_log_likelihood_time_dev / total_num_of_events_dev, 4
        )
        log_likelihood_type_final = round(
            total_log_likelihood_type_dev / total_num_of_events_dev, 4
        )
        #
        dev_end = time.time()
        #
        # start saving stuff !!!
        dict_results = {
            'log_likelihood': {
                'seq': log_likelihood_final,
                'type': log_likelihood_type_final,
                'time': log_likelihood_time_final
            },
            'model': input_test['path_pre_train'],
            'data': input_test['path_rawdata'],
            'tag_split': tag_split,
            'lists': {
                'log_likelihood_seq': list_log_likelihood_seq,
                'log_likelihood_type': list_log_likelihood_type,
                'log_likelihood_time': list_log_likelihood_time,
                'num_of_events': list_num_of_events
            }
        }
        #
        #TODO: add more info about this model
        log_org = organizers.LogOrg(
            {
                'path_tracks': os.path.abspath(
                    input_test['path_logs']
                )
            }
        )
        log_org.read_logs()
        dict_log_org = log_org.get_one_log()
        dict_results['log_info'] = dict_log_org
        #
        #
        with open(input_test['file_to_save_results'], 'wb') as f:
            pickle.dump(dict_results, f)
        #
        #
        print "the model is : ", input_test['path_pre_train']
        print "the dataset is : ", input_test['path_rawdata']
        print "the tag split is : ", tag_split
        print "the log-likelihood of seq is : ", log_likelihood_final
        print "the log-likelihood of type is : ", log_likelihood_type_final
        print "the log-likelihood of time is : ", log_likelihood_time_final
        #
        print "the three values in the order of seq / type / time : is " + str(log_likelihood_final) + ' / ' + str(log_likelihood_type_final) + ' / ' + str(log_likelihood_time_final)
        #
    print "finish testing and saving "
#
#
#
#
# Neural Hawkes process -- effect estimated by LSTM
#
# time tag means : encode time with neural networks
#
#
#
#
def train_neural_hawkes_ctsm_time(input_train):
    '''
    this function is called to train neural hawkes ctsm
    with time values fed in as features
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_train['seed_random']
    )
    #
    #command_mkdir = 'mkdir -p ' + os.path.abspath(
    #    input_train['save_file_path']
    #)
    #os.system(command_mkdir)
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': input_train['save_file_path'],
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        'min_dev_error_rate': 1e6,
        'min_dev_rmse': 1e6,
        #
        'what_to_track': input_train['what_to_track'],
        #
        'args': input_train['args'],
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        #
        'tracked_best': {},
        #
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_train['path_rawdata'],
            'size_batch': input_train['size_batch'],
            'ratio_train': input_train['ratio_train'],
            'to_read': [
                'train', 'dev'
            ],
            'partial_predict': input_train['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_train['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    #
    print "get time quantiles ... "
    data_process.get_time_quantiles()
    #
    model_settings = {
        'model': input_train['model'],
        'dim_process': data_process.dim_process,
        #
        'dim_time': data_process.dim_time,
        'dim_model': input_train['dim_model'],
        #
        'coef_l2': input_train['coef_l2'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train'],
        'threshold_time': numpy.copy(
            data_process.time_quantiles
        ),
        'predict_lambda': input_train['predict_lambda']
    }
    #
    control = controllers.ControlNeuralHawkesCTSM_time(
        model_settings
    )
    #
    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        total_log_likelihood = 0.0
        total_log_likelihood_time = 0.0
        total_log_likelihood_type = 0.0
        total_num_of_events = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                tag_batch = 'train',
                idx_batch_current = step_train,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_train['multiple_sample_for_train']
                ),
                predict_first = input_train['predict_first']
            )
            #
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_learn(
                #data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                #data_process.seq_time_rep_numpy,
                data_process.seq_time_values_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_index_in_hidden_numpy,
                data_process.seq_sims_mask_numpy
            )
            #
            #
            log_dict['iteration'] += 1
            #
            total_log_likelihood += log_likelihood_numpy
            total_log_likelihood_time += log_likelihood_time_numpy
            total_log_likelihood_type += log_likelihood_type_numpy
            total_num_of_events += num_of_events_numpy
            #
            #
            log_dict['tracked']['train_log_likelihood'] = round(
                total_log_likelihood / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_time'] = round(
                total_log_likelihood_time / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type'] = round(
                total_log_likelihood_type / total_num_of_events, 4
            )
            train_end = time.time()
            #
            log_dict['tracked']['train_time'] = round(
                (train_end - train_start)*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: go through the dev data and calculate the dev metrics
                print "Now we start validating after batches ", log_dict['track_period']
                dev_start = time.time()
                #
                #TODO: get the dev loss values
                total_log_likelihood_dev = 0.0
                total_log_likelihood_time_dev = 0.0
                total_log_likelihood_type_dev = 0.0
                total_num_of_events_dev = 0.0
                for step_dev in range(data_process.max_nums['dev']):
                    #
                    data_process.process_data(
                        tag_batch = 'dev',
                        idx_batch_current = step_dev,
                        tag_model = 'neural',
                        multiple = numpy.int32(
                            input_train[
                                'multiple_sample_for_dev'
                            ]
                        ),
                        predict_first = input_train['predict_first']
                    )
                    #
                    #print "validating ... "
                    log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                        #data_process.seq_time_to_end_numpy,
                        data_process.seq_time_to_current_numpy,
                        data_process.seq_type_event_numpy,
                        #data_process.seq_time_rep_numpy,
                        data_process.seq_time_values_numpy,
                        data_process.time_since_start_to_end_numpy,
                        data_process.num_sims_start_to_end_numpy,
                        data_process.seq_mask_numpy,
                        data_process.seq_sims_time_to_current_numpy,
                        data_process.seq_sims_index_in_hidden_numpy,
                        data_process.seq_sims_mask_numpy
                    )
                    #
                    total_log_likelihood_dev += log_likelihood_numpy
                    total_log_likelihood_time_dev += log_likelihood_time_numpy
                    total_log_likelihood_type_dev += log_likelihood_type_numpy
                    total_num_of_events_dev += num_of_events_numpy
                    #
                    if step_dev % 10 == 9:
                        print "in dev, the step is out of ", step_dev, data_process.max_nums['dev']
                #
                #
                log_dict['tracked']['dev_log_likelihood'] = round(
                    total_log_likelihood_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_time'] = round(
                    total_log_likelihood_time_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_type'] = round(
                    total_log_likelihood_type_dev / total_num_of_events_dev, 4
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round( dev_end - dev_start, 0 )
                #
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                #
                if log_dict['tracked']['dev_log_likelihood'] > log_dict['max_dev_log_likelihood']:
                    #
                    #name_file = 'model_'+str(log_dict['tracked']['track_cnt'])+'.pkl'
                    name_file = 'model.pkl'
                    save_file = os.path.abspath(
                        log_dict['save_file_path']
                    ) + '/'+name_file
                    #
                    control.save_model(save_file)
                #
                data_process.track_log(log_dict)
            ########
    data_process.finish_log(log_dict)
    print "finish training"
    #
    #
#
#
#
def test_neural_hawkes_ctsm_and_save_time(input_test):
    '''
    this function is called to test neural hawkes ctsm with time encoder
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_test['seed_random']
    )
    #
    log_dict = {
        'log_file': None,
        'save_file_path': None,
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        #
        'args': input_test['args'],
        #
        'iteration': 0,
        'track_period': None,
        'max_epoch': input_test['max_epoch'],
        'size_batch': input_test['size_batch'],
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_test['path_rawdata'],
            'size_batch': input_test['size_batch'],
            'ratio_train': numpy.float32(0.0),
            'to_read': [input_test['tag_split']],
            'partial_predict': input_test['partial_predict']
        }
    )
    #
    prune_stream = numpy.int32(
        input_test['prune_stream']
    )
    assert(prune_stream >= 0)
    if prune_stream > 0:
        data_process.prune_stream(prune_stream)
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'model': input_test['model'],
        'dim_process': data_process.dim_process,
        #
        #'dim_time': data_process.dim_time,
        #'dim_model': input_test['dim_model'],
        #
        'coef_l2': input_test['coef_l2'],
        'size_batch': input_test['size_batch'],
        'optimizer': input_test['optimizer'],
        'path_pre_train': input_test['path_pre_train'],
        'predict_lambda': input_test['predict_lambda']
    }

    control = controllers.ControlNeuralHawkesCTSM_time(
        model_settings
    )

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    tag_split = input_test['tag_split']

    for epi in range(log_dict['max_epoch']):
        #
        list_log_likelihood_seq = []
        list_log_likelihood_type = []
        list_log_likelihood_time = []
        list_num_of_events = []
        #
        dev_start = time.time()
        #
        print "validating for ", tag_split
        #
        total_log_likelihood_dev = 0.0
        total_log_likelihood_time_dev = 0.0
        total_log_likelihood_type_dev = 0.0
        total_num_of_events_dev = 0.0
        #
        for step_dev in range(data_process.max_nums[tag_split] ):
            #
            data_process.process_data(
                tag_batch = tag_split,
                idx_batch_current = step_dev,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_test['multiple_sample_for_dev']
                ),
                predict_first = input_test['predict_first']
            )
            #
            #print "training ... "
            log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                #data_process.seq_time_to_end_numpy,
                data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                #data_process.seq_time_rep_numpy,
                data_process.seq_time_values_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_index_in_hidden_numpy,
                data_process.seq_sims_mask_numpy
            )
            #
            list_log_likelihood_seq.append(
                log_likelihood_numpy
            )
            list_log_likelihood_type.append(
                log_likelihood_type_numpy
            )
            list_log_likelihood_time.append(
                log_likelihood_time_numpy
            )
            list_num_of_events.append(
                num_of_events_numpy
            )
            #
            total_log_likelihood_dev += log_likelihood_numpy
            total_log_likelihood_time_dev += log_likelihood_time_numpy
            total_log_likelihood_type_dev += log_likelihood_type_numpy
            total_num_of_events_dev += num_of_events_numpy
            #
            if step_dev % 100 == 99:
                print "in validation, the step is out of ", step_dev, data_process.max_nums[tag_split]
            #
        log_likelihood_final = round(
            total_log_likelihood_dev / total_num_of_events_dev, 4
        )
        log_likelihood_time_final = round(
            total_log_likelihood_time_dev / total_num_of_events_dev, 4
        )
        log_likelihood_type_final = round(
            total_log_likelihood_type_dev / total_num_of_events_dev, 4
        )
        #
        dev_end = time.time()
        #
        # start saving stuff !!!
        dict_results = {
            'log_likelihood': {
                'seq': log_likelihood_final,
                'type': log_likelihood_type_final,
                'time': log_likelihood_time_final
            },
            'model': input_test['path_pre_train'],
            'data': input_test['path_rawdata'],
            'tag_split': tag_split,
            'lists': {
                'log_likelihood_seq': list_log_likelihood_seq,
                'log_likelihood_type': list_log_likelihood_type,
                'log_likelihood_time': list_log_likelihood_time,
                'num_of_events': list_num_of_events
            }
        }
        #
        #TODO: add more info about this model
        log_org = organizers.LogOrg(
            {
                'path_tracks': os.path.abspath(
                    input_test['path_logs']
                )
            }
        )
        log_org.read_logs()
        dict_log_org = log_org.get_one_log()
        dict_results['log_info'] = dict_log_org
        #
        #
        with open(input_test['file_to_save_results'], 'wb') as f:
            pickle.dump(dict_results, f)
        #
        #
        print "the model is : ", input_test['path_pre_train']
        print "the dataset is : ", input_test['path_rawdata']
        print "the tag split is : ", tag_split
        print "the log-likelihood of seq is : ", log_likelihood_final
        print "the log-likelihood of type is : ", log_likelihood_type_final
        print "the log-likelihood of time is : ", log_likelihood_time_final
        #
        print "the three values in the order of seq / type / time : is " + str(log_likelihood_final) + ' / ' + str(log_likelihood_type_final) + ' / ' + str(log_likelihood_time_final)
        #
    print "finish testing and saving "
#
#
#
#
