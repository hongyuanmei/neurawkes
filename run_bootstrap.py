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
import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import modules.utils as utils
import modules.models as models
import modules.optimizers as optimizers
import modules.controllers as controllers
import modules.data_processers as data_processers
import modules.testers as testers

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
        '-fr', '--FileResults', required=True,
        help = 'Which result file to use (e.g. ./results/result.pkl)?'
    )
    #
    #'''
    parser.add_argument(
        '-cl', '--ConfidenceLevel', #required=False,
        default = 0.95, type = float,
        help='For what confidence level ? '
    )
    parser.add_argument(
        '-nb', '--NumBootstrap', #required=False,
        default = 1000, type = int,
        help='Number of Bootstrapping steps ? '
    )
    #
    parser.add_argument(
        '-s', '--Seed', #required=False,
        default = 12345, type = int,
        help='Seed of random state'
    )
    parser.add_argument(
        '-pl', '--PredictLambda', #required=False,
        default = 0, type = int,
        choices = [0,1], 
        help='Predict Lambda (intensity) ? 0--False, 1--True Note: this is used ONLY in intensity evaluation'
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
    args.ConfidenceLevel = numpy.float32(args.ConfidenceLevel)
    args.NumBootstrap = numpy.int32(args.NumBootstrap)
    args.Seed = numpy.int32(args.Seed)
    if args.PredictLambda == 0:
        args.PredictLambda = False
    else:
        args.PredictLambda = True
    #
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    #
    ## show values ##
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )
    print ("Seed is : %s" % str(args.Seed) )
    print ("FileResults is : %s" % args.FileResults )
    print ("Confidence Level is : %s" % str(args.ConfidenceLevel) )
    print ("Num of Bootstrapping is : %s" % str(args.NumBootstrap))
    print ("Predict Lambda is : %s" % args.PredictLambda)
    #
    #
    dict_args = {
        'PID': id_process,
        'TIME': time_current,
        'Seed': args.Seed,
        'FileResults': os.path.abspath(args.FileResults),
        'ConfidenceLevel': args.ConfidenceLevel,
        'NumBootstrap': args.NumBootstrap,
        'PredictLambda': args.PredictLambda
    }
    #
    numpy.random.seed(dict_args['Seed'])
    #
    #
    '''
    self.path_to_results = settings['path_to_results']
    self.confidence_level = settings[
        'confidence_level'
    ]
    self.num_bootstrap = settings[
        'num_bootstrap'
    ]
    '''
    #
    settings_ci = {
        'path_to_results': os.path.abspath(args.FileResults),
        'confidence_level': args.ConfidenceLevel,
        'num_bootstrap': args.NumBootstrap,
        'predict_lambda': args.PredictLambda
    }
    #
    tester = testers.ComputeConfidenceInterval(
        settings_ci
    )
    tester.bootstrap()
    #

if __name__ == "__main__": main()
