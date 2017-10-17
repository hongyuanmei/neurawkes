# -*- coding: utf-8 -*-
# !/usr/bin/python
#
"""

generate sequences

@author: hongyuan
"""

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
#
import modules.sequence_generators as seq_gens
import datetime
#
import argparse
__author__ = 'Hongyuan Mei'

def main():

    parser = argparse.ArgumentParser(
        description='Generating sequences... '
    )
    #
    parser.add_argument(
        '-m', '--ModelGen', #required=True,
        default = 'conttime',
        type = str,
        choices = ['hawkes', 'hawkesinhib', 'conttime'],
        help='Model used to generate data'
    )
    parser.add_argument(
        '-sp', '--SetParams', #required=False,
        default = 0,
        type = int,
        choices = [0, 1],
        help='Do we set the params ? 0 -- False; 1 -- True'
    )
    parser.add_argument(
        '-st', '--SumForTime',
        default = 0, type = int, choices = [0, 1],
        help='Do we use total intensity for time sampling? 0 -- False; 1 -- True'
    )
    parser.add_argument(
        '-fp', '--FilePretrain', required=False,
        help='File of pretrained model (e.g. ./tracks/track_PID=XX_TIME=YY/model.pkl)'
    )
    #
    parser.add_argument(
        '-s', '--Seed', #required=False,
        default = 12345, type = int,
        help='Seed of random state'
    )
    parser.add_argument(
        '-k', '--DimProcess', #required=False,
        default = 5, type = int,
        help='Number of event types'
    )
    parser.add_argument(
        '-d', '--DimLSTM', #required=False,
        default = 32, type = int,
        help='Dimension of LSTM generator'
    )
    parser.add_argument(
        '-N', '--NumSeqs', #required=False,
        default = 12000, type = int,
        help='Number of sequences to simulate'
    )
    parser.add_argument(
        '-min', '--MinLen', #required=False,
        default = 20, type = int,
        help='Min len of sequences '
    )
    parser.add_argument(
        '-max', '--MaxLen', #required=False,
        default = 100, type = int,
        help='Max len of sequences '
    )
    #
    args = parser.parse_args()
    args.DimProcess = numpy.int32(args.DimProcess)
    args.DimLSTM = numpy.int32(args.DimLSTM)
    args.Seed = numpy.int32(args.Seed)
    args.NumSeqs = numpy.int32(args.NumSeqs)
    args.MinLen = numpy.int32(args.MinLen)
    args.MaxLen = numpy.int32(args.MaxLen)
    if args.SetParams == 0:
        args.SetParams = False
    else:
        args.SetParams = True
    #
    if args.SumForTime == 0:
        args.SumForTime = False
    else:
        args.SumForTime = True
    #
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    #
    if args.SetParams:
        tag_model = '_ModelGen='+args.ModelGen+'_SetParams'+'_PID='+str(id_process)+'_TIME='+time_current
    else:
        tag_model = '_ModelGen='+args.ModelGen+'_PID='+str(id_process)+'_TIME='+time_current
    #
    #
    file_save = './data/data'+tag_model+'.pkl'
    file_save = os.path.abspath(file_save)
    file_model = './gen_models/model'+tag_model+'.pkl'
    file_model = os.path.abspath(file_model)
    #
    settings_gen = {
        'dim_process': args.DimProcess,
        'dim_LSTM': args.DimLSTM,
        #'dim_states': args.DimStates,
        'seed_random': args.Seed,
        'path_pre_train': args.FilePretrain,
        'sum_for_time': args.SumForTime,
        'args': None
    }
    settings_gen_seqs = {
        'num_seqs': args.NumSeqs,
        'min_len': args.MinLen,
        'max_len': args.MaxLen
    }
    #print settings_gen_seqs
    #
    flag_1 = (
        args.ModelGen == 'hawkes' or args.ModelGen == 'hawkesinhib' or args.ModelGen == 'neural' or args.ModelGen == 'neuralgeneral' or args.ModelGen == 'fst' or args.ModelGen == 'neuraladapt'
    )
    flag_2 = (
        args.ModelGen == 'neuraladapttime' or args.ModelGen == 'neuraladapttimescale' or args.ModelGen == 'neuralreduce' or args.ModelGen == 'conttime'
    )
    assert( flag_1 or flag_2 )
    #
    if args.ModelGen == 'hawkes':
        gen_model = seq_gens.HawkesGen(settings_gen)
    elif args.ModelGen == 'hawkesinhib':
        gen_model = seq_gens.HawkesInhibGen(settings_gen)
    elif args.ModelGen == 'neural':
        gen_model = seq_gens.NeuralHawkesGen(settings_gen)
    elif args.ModelGen == 'neuralgeneral':
        gen_model = seq_gens.GeneralizedNeuralHawkesGen(
            settings_gen
        )
    elif args.ModelGen == 'neuraladapt':
        gen_model = seq_gens.NeuralHawkesAdaptiveBaseGen(
            settings_gen
        )
    elif args.ModelGen == 'neuraladapttime':
        gen_model = seq_gens.NeuralHawkesAdaptiveBaseGen_time(
            settings_gen
        )
    elif args.ModelGen == 'neuraladapttimescale':
        gen_model = seq_gens.NeuralHawkesAdaptiveBaseGen_time_scale(
            settings_gen
        )
    elif args.ModelGen == 'neuralreduce':
        gen_model = seq_gens.NeuralHawkesAdaptiveBaseGen_time_scale_reduce(
            settings_gen
        )
    elif args.ModelGen == 'conttime':
        gen_model = seq_gens.NeuralHawkesCTLSTM(
            settings_gen
        )
    #elif args.ModelGen == 'fst':
    #    gen_model = seq_gens.FSTGen(settings_gen)
    else:
        print "Generator NOT implemented : ", args.ModelGen
    #
    #
    if args.SetParams:
        gen_model.set_params()
        args.DimProcess = gen_model.dim_process
    #
    ## show values ##
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )
    print ("Seed is : %s" % str(args.Seed) )
    print ("FilePretrain is : %s" % args.FilePretrain)
    print ("Generator is : %s" % args.ModelGen )
    print ("SetParams is : %s" % args.SetParams )
    print ("FileSave is : %s" % file_save )
    print ("FileModel is : %s" % file_model )
    print ("DimProcess is : %s" % str(args.DimProcess) )
    if 'neural' in args.ModelGen or 'conttime' in args.ModelGen:
        print ("DimLSTM is : %s" % str(args.DimLSTM) )
    #if 'fst' in args.ModelGen:
    #    print ("DimStates is : %s" % str(args.DimStates) )
    print ("NumSeqs is : %s" % str(args.NumSeqs) )
    print ("MinLen is : %s" % str(args.MinLen) )
    print ("MaxLen is : %s" % str(args.MaxLen) )
    print ("SumForTime is : %s" % str(args.SumForTime) )
    #
    #
    dict_args = {
        'PID': id_process,
        'TIME': time_current,
        'ModelGen': args.ModelGen,
        'SetParams': args.SetParams,
        'FileSave': file_save,
        'FileModel': file_model,
        'DimProcess': args.DimProcess,
        'DimLSTM': args.DimLSTM,
        #'DimStates': args.DimStates,
        'Seed': args.Seed,
        'FilePretrain': args.FilePretrain,
        'NumSeqs': args.NumSeqs,
        'MinLen': args.MinLen,
        'MaxLen': args.MaxLen,
        'SumForTime': args.SumForTime
    }
    #
    gen_model.set_args(dict_args)
    #
    cut_train = numpy.int32(8000)
    cut_dev = numpy.int32(9000)
    cut_test = numpy.int32(10000)
    print "The cut off for training, dev, test and test1 are : ", (cut_train, cut_dev, cut_test, args.NumSeqs)
    #
    #
    time_0 = time.time()
    gen_model.gen_seqs(settings_gen_seqs)
    time_1 = time.time()
    dtime = time_1 - time_0
    gen_model.print_some()
    #
    gen_model.save_model(file_model)
    #
    dict_data = {
        'train': gen_model.list_seqs[:cut_train],
        'dev': gen_model.list_seqs[cut_train:cut_dev],
        'test': gen_model.list_seqs[
            cut_dev:cut_test
        ],
        'test1': gen_model.list_seqs[
            cut_test:
        ],
        'args': dict_args
    }
    #
    print "saving ... "
    with open(file_save, 'wb') as f:
        pickle.dump(dict_data, f)

    print "finished ! Took {} seconds !!!".format(str(round(dtime,2)))
    #

if __name__ == "__main__": main()
