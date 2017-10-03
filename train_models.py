# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Mar 18th 10:58:37 2016

train a continuous-time sequential model

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
#import scipy.io
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
        description='Trainning model ... '
    )
    #
    parser.add_argument(
        '-m', '--Model', required=True,
        choices = ['hawkes', 'hawkesinhib', 'conttime'],
        help='Which model to train? hawkes (SE-MPP)? hawkesinhib (D-SM-MPP)? conttime (N-SM-MPP)?'
    )
    parser.add_argument(
        '-fd', '--FileData', required=True,
        help='Path of the dataset (e.g. ./data/data_hawkes/)'
    )
    #
    parser.add_argument(
        '-tr', '--TrainRatio', #required=False,
        default = 1.0, type = float,
        help='How much data to train?'
    )
    #
    parser.add_argument(
        '-cl2', '--CoefL2', #required=False,
        default = 0.0, type = float,
        help='Coefficient of L2 norm'
    )
    #
    parser.add_argument(
        '-d', '--DimLSTM', #required=False,
        default = 64, type = int,
        help='Dimension of LSTM model '
    )
    parser.add_argument(
        '-s', '--Seed', #required=False,
        default = 12345, type = int,
        help='Seed of random state'
    )
    #
    parser.add_argument(
        '-fp', '--FilePretrain', required=False,
        help='File of pretrained model (e.g. ./tracks/track_PID=XX_TIME=YY/model.pkl)'
    )
    parser.add_argument(
        '-tp', '--TrackPeriod', #required=False,
        default = 1000, type = int,
        help='Track period of training'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', #required=False,
        default = 50, type = int,
        help='Max epoch number of training'
    )
    parser.add_argument(
        '-sb', '--SizeBatch', #required=False,
        default = 10, type = int,
        help='Size of mini-batch'
    )
    parser.add_argument(
        '-op', '--Optimizer', #required=False,
        default = 'adam', type = str,
        choices = ['adam', 'sgd'],
        help='Optimizer of training'
    )
    parser.add_argument(
        '-mt', '--MultipleTrain', #required=False,
        default = 1, type = int,
        help='Multiple of events to sample (integral) for training'
    )
    parser.add_argument(
        '-md', '--MultipleDev', #required=False,
        default = 10, type = int,
        help='Multiple of events to sample (integral) for dev'
    )
    parser.add_argument(
        '-wt', '--WhatTrack', #required=False,
        default = 'loss', type = str,
        choices = ['loss', 'rmse', 'rate'],
        help='What to track for early stoping ? '
    )
    parser.add_argument(
        '-ls', '--LossType', #required=False,
        default = 'loglikehood', type = str,
        choices = ['loglikehood', 'prediction'],
        help='What is the loss to optimized ?'
    )
    parser.add_argument(
        '-lr', '--LearnRate', #required=False,
        default = 1e-3, type = float,
        help='What learning rate to use ?'
    )
    parser.add_argument(
        '-pp', '--PartialPredict', #required=False,
        default = 0, type = int,
        choices = [0, 1],
        help='What to only predict part of stream ? 0--False, 1--True'
    )
    parser.add_argument(
        '-ps', '--PruneStream', #required=False,
        default = 0, type = int,
        help='Prune stream? Give me the index ! 0 is nothng to prune. Note : index specifies a COMBINATION of event types by its binary coding (e.g. 0--00000, 1--00001, 31-11111 where 1 means this type is pruned)!'
    )
    parser.add_argument(
        '-ds', '--DevIncludedSetting',#required=False,
        default = 0, type = int,
        choices = [0,1],
        help='Alternative setting (fix tuned hyper-params, train on combo of train and dev, then test)? 0--False, 1--True Note: in our project, this is ONLY used to compare prev work on MIMIC, SO and Financial datasets'
    )
    parser.add_argument(
        '-pf', '--PredictFirst', #required=False,
        default = 1, type = int,
        choices = [0,1],
        help='Predict the first event ? 0--False, 1--True Note: in our project, this is False ONLY on MIMIC, SO and Financial datasets'
    )
    parser.add_argument(
        '-pl', '--PredictLambda', #required=False,
        default = 0, type = int,
        choices = [0,1],
        help='Predict Lambda (intensity) ? 0--False, 1--True Note: this is used ONLY in intensity evaluation'
    )
    '''
    They train model on entire training and eval on test after training, i.e., no dev/validation set
    We only use this setting when compared with them on their dataset
    Otherwise, we use dev/validation set to tune params and early stop, and only eval on test after the model is fixed.
    '''
    #
    #
    args = parser.parse_args()
    #
    #
    args.TrainRatio = numpy.float32(args.TrainRatio)
    assert(args.TrainRatio > 0.0 and args.TrainRatio <= 1.0)
    #
    args.CoefL2 = numpy.float32(args.CoefL2)
    assert(args.CoefL2 >= 0.0)
    args.DimLSTM = numpy.int32(args.DimLSTM)
    args.Seed = numpy.int32(args.Seed)
    args.TrackPeriod = numpy.int32(args.TrackPeriod)
    args.MaxEpoch = numpy.int32(args.MaxEpoch)
    args.SizeBatch = numpy.int32(args.SizeBatch)
    args.MultipleTrain = numpy.int32(args.MultipleTrain)
    args.MultipleDev = numpy.int32(args.MultipleDev)
    #
    if args.LossType == 'prediction':
        assert(args.WhatTrack == 'rmse' or args.WhatTrack == 'rate')
    else:
        assert(args.WhatTrack == 'loss')
    #
    args.LearnRate = numpy.float32(args.LearnRate)
    assert(args.LearnRate > 0.0)
    #
    if args.PartialPredict == 0:
        args.PartialPredict = False
    else:
        args.PartialPredict = True
    #
    args.PruneStream = numpy.int32(args.PruneStream)
    #
    if args.DevIncludedSetting == 0:
        args.DevIncludedSetting = False
    else:
        args.DevIncludedSetting = True
    #
    if args.PredictFirst == 0:
        args.PredictFirst = False
    else:
        args.PredictFirst = True
    #
    if args.PredictLambda == 0:
        args.PredictLambda = False
    else:
        args.PredictLambda = True
    #
    #
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    #
    flag_1 = (
        args.Model == 'hawkes' or args.Model == 'hawkesinhib' or args.Model == 'neural' or args.Model == 'neuralgeneral' or args.Model == 'neuraladapt' or args.Model == 'neuraltime' or args.Model == 'neuralgeneraltime' or args.Model == 'neuraladapttime'
    )
    flag_2 = (
        args.Model == 'nanmodel'
    )
    flag_3 = (
        args.Model == 'neuraladapttimescale' or args.Model == 'hawkesinhibscale' or args.Model == 'neuralreduce' or args.Model == 'conttime'
    )
    #
    # conttime is the one with continuous time LSTM
    #
    assert(flag_1 or flag_2 or flag_3)

    # we stop using neuralsimple
    # +time means we encode time using neural networks
    #
    tag_model = '_PID='+str(id_process)+'_TIME='+time_current
    #
    #file_log = os.path.abspath(
    #    './logs/log' + tag_model + '.txt'
    #)
    #path_save = os.path.abspath(
    #    './models/models' + tag_model + '/'
    #)
    if 'meme' in args.FileData:
        tag_track = '_meme'
    elif 'retweet' in args.FileData:
        tag_track = '_retweet'
    elif 'mimic' in args.FileData:
        tag_track = '_mimic'
    elif '_so' in args.FileData:
        tag_track = '_so'
    elif '_bookorder' in args.FileData:
        tag_track = '_bookorder'
    elif '_missing' in args.FileData:
        tag_track = '_missing'
    else:
        tag_track = ''
    #
    path_track = './tracks'+ tag_track +'/track' + tag_model + '/'
    file_log = os.path.abspath(
        path_track + 'log.txt'
    )
    #path_save = os.path.abspath(
    #    path_track + 'models/'
    #)
    path_save = path_track
    #
    command_mkdir = 'mkdir -p ' + os.path.abspath(
        path_track
    )
    os.system(command_mkdir)
    #
    #
    ## show values ##
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )
    print ("Seed is : %s" % str(args.Seed) )
    #
    print ("Model is : %s" % args.Model )
    print ("CoefL2 is : %s" % str(args.CoefL2) )
    print ("FileData is : %s" % args.FileData )
    print ("TrainRatio is : %s" % str(args.TrainRatio) )
    if 'neural' in args.Model or 'nanmodel' in args.Model:
        print ("DimLSTM is : %s" % str(args.DimLSTM) )
    print ("FilePretrain is : %s" % args.FilePretrain)
    print ("TrackPeriod is : %s" % str(args.TrackPeriod) )
    print ("MaxEpoch is : %s" % str(args.MaxEpoch) )
    print ("SizeBatch is : %s" % str(args.SizeBatch) )
    print ("Optimizer is : %s" % args.Optimizer)
    print ("LossType is : %s" % args.LossType)
    print ("WhatTrack is : %s" % args.WhatTrack)
    print ("LearnRate is : %s" % args.LearnRate)
    print ("PartialPredict is : %s" % args.PartialPredict)
    print ("PruneStream is : %s" % str(args.PruneStream) )
    print ("Dev Included Setting is: %s" % args.DevIncludedSetting )
    print ("PredictFirst is: %s" % args.PredictFirst )
    print ("PredictLambda is: %s" % args.PredictLambda )
    #
    flag_show_1 = (
        args.Model == 'hawkesinhib' or args.Model == 'neural' or args.Model == 'neuralgeneral' or args.Model == 'neuraladapt' or args.Model == 'neuralsimple' or args.Model == 'neuraltime' or args.Model == 'neuralgeneraltime' or args.Model == 'neuraladapttime'
    )
    flag_show_2 = (
        args.Model == 'hawkesinhibscale' or args.Model == 'neuraladapttimescale' or args.Model == 'neuralreduce' or args.Model == 'conttime'
    )
    #
    if (flag_show_1 and flag_show_2):
        print ("Multiple for training is : %s" % args.MultipleTrain)
        print ("Multiple for dev is : %s" % args.MultipleDev)
    #
    dict_args = {
        'PID': id_process,
        'TIME': time_current,
        'Seed': args.Seed,
        #
        'Model': args.Model,
        'CoefL2': args.CoefL2,
        'FileData': args.FileData,
        'TrainRatio': args.TrainRatio,
        'DimLSTM': args.DimLSTM,
        'FilePretrain': args.FilePretrain,
        'TrackPeriod': args.TrackPeriod,
        'MaxEpoch': args.MaxEpoch,
        'SizeBatch': args.SizeBatch,
        'Optimizer': args.Optimizer,
        'MultipleTrain': args.MultipleTrain,
        'MultipleDev': args.MultipleDev,
        'LossType': args.LossType,
        'WhatTrack': args.WhatTrack,
        'LearnRate': args.LearnRate,
        'PartialPredict': args.PartialPredict,
        'PruneStream': args.PruneStream,
        'DevIncludedSetting': args.DevIncludedSetting,
        'PredictLambda': args.PredictLambda
    }
    #
    input_train = {
        'model': args.Model,
        'seed_random': args.Seed,
        'path_rawdata': args.FileData,
        'ratio_train': args.TrainRatio,
        'path_pre_train': args.FilePretrain,
        'track_period': args.TrackPeriod,
        'max_epoch': args.MaxEpoch,
        'size_batch': args.SizeBatch,
        'dim_model': args.DimLSTM,
        'optimizer': args.Optimizer,
        'save_file_path': path_save,
        'log_file': file_log,
        'args': dict_args,
        'coef_l2': args.CoefL2,
        'what_to_track': args.WhatTrack,
        'loss_type': args.LossType,
        'learn_rate': args.LearnRate,
        'partial_predict': args.PartialPredict,
        'prune_stream': args.PruneStream,
        'di_setting': args.DevIncludedSetting,
        'predict_lambda': args.PredictLambda
    }
    #
    if '_so' in args.FileData or '_mimic' in args.FileData or '_bookorder' in args.FileData:
        input_train['predict_first'] = False
    else:
        if args.PredictFirst:
            input_train['predict_first'] = True
        else:
            input_train['predict_first'] = False
    #
    #
    flag_multiple_1 = (
        args.Model == 'hawkesinhib' or args.Model == 'neural' or args.Model == 'neuralgeneral' or args.Model == 'neuraladapt' or args.Model == 'neuralsimple' or args.Model == 'neuraltime' or args.Model == 'neuralgeneraltime' or args.Model == 'neuraladapttime'
    )
    flag_multiple_2 = (
        args.Model == 'hawkesinhibscale' or args.Model == 'neuraladapttimescale' or args.Model == 'neuralreduce' or args.Model == 'conttime'
    )
    #
    if (flag_multiple_1 or flag_multiple_2):
        input_train['multiple_sample_for_train'] = numpy.int32(
            args.MultipleTrain
        )
        input_train['multiple_sample_for_dev'] = numpy.int32(
            args.MultipleDev
        )
    #
    if args.Model == 'hawkes':
        run_models.train_hawkes_ctsm(input_train)
    elif args.Model == 'hawkesinhib' or args.Model == 'hawkesinhibscale':
        run_models.train_hawkesinhib_ctsm(input_train)
    elif args.Model == 'neural':
        run_models.train_neural_hawkes_ctsm(input_train)
    elif args.Model == 'neuralgeneral':
        run_models.train_generalized_neural_hawkes_ctsm(
            input_train, tag_neural_type = 'general'
        )
    elif args.Model == 'neuraladapt':
        run_models.train_generalized_neural_hawkes_ctsm(
            input_train, tag_neural_type = 'adaptive'
        )
    elif args.Model == 'neuralsimple':
        run_models.train_generalized_neural_hawkes_ctsm(
            input_train, tag_neural_type = 'simple'
        )
    elif args.Model == 'neuraltime':
        run_models.train_neural_hawkes_ctsm_time(
            input_train
        )
    elif args.Model == 'neuralgeneraltime':
        run_models.train_generalized_neural_hawkes_ctsm_time(
            input_train, tag_neural_type = 'general'
        )
    elif args.Model == 'neuraladapttime' or args.Model == 'neuraladapttimescale' or args.Model == 'neuralreduce' or args.Model == 'conttime':
        if args.DevIncludedSetting:
            run_models.train_generalized_neural_hawkes_ctsm_time_DevIncludedSetting(
                input_train, tag_neural_type = 'adaptive'
            )
        else:
            run_models.train_generalized_neural_hawkes_ctsm_time(
                input_train, tag_neural_type = 'adaptive'
            )
    else:
        print "Model not implemented yet !!! "
    #

if __name__ == "__main__": main()
