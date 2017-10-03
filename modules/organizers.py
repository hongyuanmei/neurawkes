# -*- coding: utf-8 -*-
"""

Here are the organizers
to organize the project folder

@author: hongyuan
"""

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import fnmatch
#import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import utils
import csv

dtype=theano.config.floatX

class LogOrg(object):
    #
    def __init__(self, settings):
        print "log files organizer ... "
        self.path_tracks = settings['path_tracks']
        if self.path_tracks == None:
            self.path_tracks = os.path.abspath(
                './tracks/'
            )
        else:
            self.path_tracks = os.path.abspath(
                self.path_tracks
            )
        #
        self.list_logs = None
        #
    #
    def parse_log(self, path_log):
        with open(path_log, 'r') as f:
            str_log = f.read()
        #
        if 'The best model info is' not in str_log:
            print "Here is a unfinished training : "
            print path_log
            #TODO: the training is not finished
            #so we can grab the best info as the current best
            segment_args = str_log.split(
                'Before training, the compilation time is '
            )[0].split(
                'Model specs are listed below : '
            )[-1]
            #
            segment_content = str_log.split("\n\n")[::-1]
            #
            content_with_best = ''
            idx = 0
            while 'a new best model' not in content_with_best:
                content_with_best = segment_content[idx]
                idx += 1
            #
            segment_best = content_with_best
            #
        else:
            #TODO: the training is finished
            #so we can grab the best info at the bottom
            segment_args = str_log.split(
                'Before training, the compilation time is '
            )[0].split(
                'Model specs are listed below : '
            )[-1]
            segment_best = str_log.split(
                'The best model info is shown below : '
            )[-1]
            #print segment_args
            #print segment_best
        dict_args = {}
        dict_args['PathTrack'] = os.path.dirname(path_log)
        #
        lines_args = segment_args.split('\n')
        for line in lines_args:
            if line != '\n' and line != '':
                #print line
                list_line = line.split(' : ')
                assert(len(list_line)==2 )
                dict_args[list_line[0]] = list_line[-1]
        #
        lines_best = segment_best.split('\n')
        for line in lines_best:
            if line != '\n' and line != '' and 'dev_' in line:
                #print line
                list_line = line.split(' is ')
                #print path_log
                #print list_line
                #print segment_best
                assert(len(list_line)==2 )
                dict_args[list_line[0]] = list_line[-1]
        #
        return dict_args
        #
    #
    def read_logs(self):
        self.list_logs = []
        #
        print "reading and parsing all the log files ... "
        #
        for path, dirs, files in os.walk(self.path_tracks):
            for name_file in fnmatch.filter(files, 'log.txt'):
                path_log = os.path.abspath(
                    os.path.join(path, name_file)
                )
                #
                self.list_logs.append(
                    self.parse_log(path_log)
                )
                #
            #
        #
        print "done ! "
        #
    #
    #
    def get_one_log(self, idx_log=0):
        print "getting one log idx : ", idx_log
        assert(len(self.list_logs)>0 )
        #path_save = os.path.abspath(path_save)
        #path_save = path_save + '/' + 'logs.csv'
        dict_args = self.list_logs[idx_log]
        return dict_args
    #
    def save_csv(self, path_save):
        print "saving logs into a csv ... "
        assert(len(self.list_logs)>0 )
        path_save = os.path.abspath(path_save)
        #path_save = path_save + '/' + 'logs.csv'
        #
        names_field = []
        for dict_args in self.list_logs:
            for the_key in dict_args:
                if the_key not in names_field:
                    names_field.append(the_key)
        names_field.sort()
        #
        with open(path_save, 'w') as file_csv:
            writer_csv = csv.DictWriter(
                file_csv, fieldnames = names_field
            )
            writer_csv.writeheader()
            for dict_args in self.list_logs:
                writer_csv.writerow(
                    dict_args
                )
            #
        print "done ! "
        #
    #
#
#

def main():
    log_org = LogOrg(
        {
            'path_tracks': '../tracks/'
        }
    )
    log_org.read_logs()
    log_org.save_csv(
        '../'
    )

if __name__ == "__main__": main()
