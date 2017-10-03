# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Mar 18th 10:58:37 2016

test a continuous-time sequential model

@author: hongyuan
"""

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import sys
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import modules.utils as utils
import modules.models as models
import modules.optimizers as optimizers
import modules.controllers as controllers
import modules.data_processers as data_processers

import run_models
import datetime

dtype=theano.config.floatX


#
import argparse
__author__ = 'Hongyuan Mei'

def main():

    parser = argparse.ArgumentParser(
        description='Testing model ... '
    )
    #
    parser.add_argument(
        '-m', '--Model', required=True,
        choices = ['hawkes', 'hawkesinhib', 'conttime'],
        help='Which model to test? hawkes (SE-MPP)? hawkesinhib (D-SM-MPP)? conttime (N-SM-MPP)?'
    )
    parser.add_argument(
        '-fd', '--FileData', required=True,
        help='Path of the dataset (e.g. ./data/data_hawkes/)'
    )
    parser.add_argument(
        '-fp', '--FilePretrain', required=True,
        help='File of pretrained model (e.g. ./tracks/track_PID=XX_TIME=YY/model.pkl)'
    )
    parser.add_argument(
        '-ts', '--TagSplit', required=True,
        choices = ['dev', 'test', 'test1'],
        help='Which split to test? Dev or Test?'
    )
    #
    parser.add_argument(
        '-s', '--Seed', #required=False,
        default = 12345, type = int,
        help='Seed of random state'
    )
    parser.add_argument(
        '-md', '--MultipleDev', #required=False,
        default = 10, type = int,
        help='Multiple of events to sample (integral) for dev/test'
    )
    parser.add_argument(
        '-sl', '--SaveLog', #required=False,
        default = 1, type = int,
        choices = [0,1],
        help='Do you want to save the log ? 0--False, 1--True'
    )
    parser.add_argument(
        '-pp', '--PartialPredict', #required=False,
        default = 0, type = int,
        choices = [0,1],
        help='What to only predict part of stream ? 0--False, 1--True'
    )
    parser.add_argument(
        '-ps', '--PruneStream', #required=False,
        default = 0, type = int,
        help='Prune stream? Give me the index ! 0 is nothng to prune. Note : index specifies a COMBINATION of event types by its binary coding (e.g. 0--00000, 1--00001, 31-11111 where 1 means this type is pruned)!'
    )
    parser.add_argument(
        '-pf', '--PredictFirst', #required=False,
        default = 1, type = int,
        choices = [0,1],
        help='Predict the first event ? 0--False, 1--True Note: in our project, this is False ONLY on MIMIC, SO and Financial datasets'
    )
    #
    #TODO: related to intensity eval
    parser.add_argument(
        '-pl', '--PredictLambda', #required=False,
        default = 0, type = int,
        choices = [0,1],
        help='Predict Lambda (intensity) ? 0--False, 1--True Note: this is used ONLY in intensity evaluation'
    )
    parser.add_argument(
        '-fg', '--FileGold', required=False,
        help='Gold model to eval intensity ? (e.g. ./gold_models/model.pkl)'
    )
    parser.add_argument(
        '-mg', '--ModelGold', required=False,
        choices = ['hawkes', 'hawkesinhib', 'conttime'],
        help='Gold Model to be used '
    )
    #
    args = parser.parse_args()
    #
    #if args.TrackPeriod == None:
    #    args.TrackPeriod = numpy.int32(100)
    #else:
    #    args.TrackPeriod = numpy.int32(args.TrackPeriod)
    #
    #
    args.Seed = numpy.int32(args.Seed)
    args.MultipleDev = numpy.int32(args.MultipleDev)
    if args.SaveLog == 0:
        args.SaveLog = False
    else:
        args.SaveLog = True
    #
    if args.PartialPredict == 0:
        args.PartialPredict = False
    else:
        args.PartialPredict = True
    #
    args.PruneStream = numpy.int32(args.PruneStream)
    #
    if args.PredictFirst == 0:
        args.PredictFirst = False
    else:
        args.PredictFirst = True
    #
    # For intensity eval
    #
    if args.PredictLambda == 0:
        args.PredictLambda = False
    else:
        args.PredictLambda = True
    #
    if args.FileGold == None:
        assert(args.PredictLambda==False)
    else:
        args.FileGold = str(args.FileGold)
    if args.ModelGold == None:
        assert(args.PredictLambda==False)
    else:
        args.ModelGold = str(args.ModelGold)
    #
    #
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    #
    ## show values ##
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )
    print ("Model is : %s" % args.Model )
    #print ("CoefL2 is : %s" % str(args.CoefL2) )
    print ("FileData is : %s" % args.FileData )
    #if 'lstm' in args.Model:
    #    print ("DimLSTM is : %s" % str(args.DimLSTM) )
    print ("Seed is : %s" % str(args.Seed) )
    print ("FilePretrain is : %s" % args.FilePretrain)
    #print ("TrackPeriod is : %s" % str(args.TrackPeriod) )
    #print ("MaxEpoch is : %s" % str(args.MaxEpoch) )
    print ("SizeBatch is : %s" % str(1) )
    print ("PartialPredict is : %s" % args.PartialPredict)
    print ("PruneStream is : %s" % str(args.PruneStream) )
    print ("PredictFirst is: %s" % args.PredictFirst )
    print ("PredictLambda is : %s" % str(args.PredictLambda) )
    print ("ModelGold is : %s" % args.ModelGold )
    print ("FileGold is : %s" % args.FileGold )
    #print ("Optimizer is : %s" % args.Optimizer)
    flag_show_1 = (
        args.Model == 'hawkesinhib' or args.Model == 'neural' or args.Model == 'neuralgeneral' or args.Model == 'neuraladapt' or args.Model == 'neuralsimple' or args.Model == 'neuraltime' or args.Model == 'neuralgeneraltime' or args.Model == 'neuraladapttime'
    )
    flag_show_2 = (
        args.Model == 'hawkesinhibscale' or args.Model == 'neuraladapttimescale' or args.Model == 'neuralreduce' or args.Model == 'conttime'
    )
    #
    if (flag_show_1 and flag_show_2):
        print ("Multiple for testing is : %s" % args.MultipleDev)
    #
    #
    dict_args = {
        'PID': id_process,
        'TIME': time_current,
        'Model': args.Model,
        #'CoefL2': args.CoefL2,
        'FileData': args.FileData,
        #'DimLSTM': args.DimLSTM,
        'Seed': args.Seed,
        'FilePretrain': args.FilePretrain,
        #'TrackPeriod': args.TrackPeriod,
        #'MaxEpoch': args.MaxEpoch,
        'SizeBatch': numpy.int32(1),
        #'ConfidenceLevel': args.ConfidenceLevel,
        #'NumBootstrap': args.NumBootstrap,
        #'Optimizer': args.Optimizer,
        #'MultipleTrain': args.MultipleTrain,
        'MultipleDev': args.MultipleDev,
        'PartialPredict': args.PartialPredict,
        'PruneStream': args.PruneStream,
        'PredictLambda': args.PredictLambda,
        'ModelGold': args.ModelGold,
        'FileGold': args.FileGold
    }
    #
    #some unuseful arguments are set to dummy values here
    #like optimizer, max_epoch, size_batch
    input_test = {
        'model': args.Model,
        'seed_random': args.Seed,
        'path_rawdata': args.FileData,
        'path_pre_train': args.FilePretrain,
        'track_period': None,
        'max_epoch': numpy.int32(1),
        'size_batch': numpy.int32(1),
        #'dim_model': args.DimLSTM,
        'optimizer': 'adam',
        'save_file_path': None,
        'log_file': None,
        'tag_split': args.TagSplit,
        'args': dict_args,
        'coef_l2': numpy.float32(0.0),
        #'cl': args.ConfidenceLevel,
        #'num_bootstrap': args.NumBootstrap,
        'loss_type': 'loglikehood',
        'partial_predict': args.PartialPredict,
        'prune_stream': args.PruneStream,
        'predict_lambda': args.PredictLambda,
        'path_gold': args.FileGold,
        'model_gold': args.ModelGold
    }
    #
    if '_so' in args.FileData or '_mimic' in args.FileData or '_bookorder' in args.FileData:
        input_test['predict_first'] = False
    else:
        if args.PredictFirst:
            input_test['predict_first'] = True
        else:
            input_test['predict_first'] = False
    #
    #
    path_pre_train = input_test['path_pre_train']
    path_with_no_name = path_pre_train.replace(
        '/'+os.path.basename(path_pre_train), ''
    )
    tag_with_track = os.path.basename(path_with_no_name)
    tag_model = tag_with_track.replace(
        'track', ''
    )
    #
    input_test['path_logs'] = path_with_no_name
    if args.SaveLog == False:
        input_test['path_logs'] = None
    # code drifts with time
    # when model is trained model,
    # tag_model is PID and TIME
    # but when model is gen_model
    # tag_model is just gen_model the string
    # so we will add _ to make the format match
    #if '_' not in tag_model:
    if tag_model[0] != '_':
        tag_model = '_' + tag_model
    #
    if '_hawkes' in input_test['path_rawdata']:
        tag_data = 'hawkes'
    if '_hawkesinhib' in input_test['path_rawdata']:
        tag_data = 'hawkesinhib'
    if '_neural' in input_test['path_rawdata']:
        tag_data = 'neural'
    if '_neuralgeneral' in input_test['path_rawdata']:
        tag_data = 'neuralgeneral'
    if '_neuraladapt' in input_test['path_rawdata']:
        tag_data = 'neuraladapt'
    if '_neuralsimple' in input_test['path_rawdata']:
        tag_data = 'neuralsimple'
    if 'data_kaggle' in input_test['path_rawdata']:
        tag_data = 'kaggle'
    if 'data_retweet' in input_test['path_rawdata']:
        tag_data = 'retweet'
    if 'data_food' in input_test['path_rawdata']:
        tag_data = 'food'
    if 'data_meme' in input_test['path_rawdata']:
        tag_data = 'meme'
    if 'data_mimic' in input_test['path_rawdata']:
        tag_data = 'mimic'
    #
    if 'data_so' in input_test['path_rawdata']:
        tag_data = 'so'
    if 'data_bookorder' in input_test['path_rawdata']:
        tag_data = 'bookorder'
    #
    if 'data_neuraladapttimescale' in input_test['path_rawdata']:
        tag_data = 'neuraladapttimescale'
    if 'data_neuralreduce' in input_test['path_rawdata']:
        tag_data = 'neuralreduce'
    if 'data_missing' in input_test['path_rawdata']:
        tag_data = 'missing'
    #
    if 'data_conttime' in input_test['path_rawdata']:
        tag_data = 'conttime'
    #
    name_model = 'results_Model='+input_test[
        'args'
    ]['Model']+'_Data='+tag_data+'_Split='+input_test[
        'tag_split'
    ]+tag_model+'.pkl'
    #
    path_to_save_results = os.path.abspath(
        './results_'+tag_data
    )
    if args.PredictLambda:
        path_to_save_results += '_predictlambda'
    file_to_save_results = path_to_save_results + '/' + name_model
    #
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)
    #
    input_test['file_to_save_results'] = file_to_save_results
    #
    flag_multiple_1 = (
        args.Model == 'hawkesinhib' or args.Model == 'neural' or args.Model == 'neuralgeneral' or args.Model == 'neuraladapt' or args.Model == 'neuralsimple' or args.Model == 'neuraltime' or args.Model == 'neuralgeneraltime' or args.Model == 'neuraladapttime'
    )
    flag_multiple_2 = (
        args.Model == 'hawkesinhibscale' or args.Model == 'neuraladapttimescale' or args.Model == 'neuralreduce' or args.Model == 'conttime'
    )
    #
    if (flag_multiple_1 or flag_multiple_2):
        input_test['multiple_sample_for_train'] = numpy.int32(1)
        input_test['multiple_sample_for_dev'] = numpy.int32(
            args.MultipleDev
        )
    #
    #
    #
    # start running
    if args.PredictLambda:
        run_models.test_intensity_and_save(input_test)
    else:
        if args.Model == 'hawkes':
            run_models.test_hawkes_ctsm_and_save(
                input_test
            )
        elif args.Model == 'hawkesinhib' or args.Model == 'hawkesinhibscale':
            run_models.test_hawkesinhib_ctsm_and_save(
                input_test
            )
            #run_models.test_hawkesinhib_ctsm_confidence_interval(input_test)
        elif args.Model == 'neural':
            run_models.test_neural_hawkes_ctsm_and_save(
                input_test
            )
            #run_models.test_neural_hawkes_ctsm_confidence_interval(input_test)
        elif args.Model == 'neuralgeneral':
            run_models.test_generalized_neural_hawkes_ctsm_and_save(
                input_test, tag_neural_type = 'general'
            )
        elif args.Model == 'neuraladapt':
            run_models.test_generalized_neural_hawkes_ctsm_and_save(
                input_test, tag_neural_type = 'adaptive'
            )
        elif args.Model == 'neuralsimple':
            run_models.test_generalized_neural_hawkes_ctsm_and_save(
                input_test, tag_neural_type = 'simple'
            )
        elif args.Model == 'neuraltime':
            run_models.test_generalized_neural_hawkes_ctsm_and_save_time(
                input_test, tag_neural_type = 'neural'
            )
        elif args.Model == 'neuralgeneraltime':
            run_models.test_generalized_neural_hawkes_ctsm_and_save_time(
                input_test, tag_neural_type = 'general'
            )
        elif args.Model == 'neuraladapttime' or args.Model == 'neuraladapttimescale' or args.Model == 'neuralreduce' or args.Model == 'conttime':
            run_models.test_generalized_neural_hawkes_ctsm_and_save_time(
                input_test, tag_neural_type='adaptive'
            )
            #if args.PredictLambda:
            #    run_models.test_generalized_neural_hawkes_intensity_and_save_time(
            #        input_test, tag_neural_type='adaptive'
            #    )
            #else:
        else:
            print "Model not implemented yet !!! "
        #
    #

if __name__ == "__main__": main()
