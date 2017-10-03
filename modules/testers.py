# -*- coding: utf-8 -*-
"""

Here are some testers used in our project 

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
import csv

dtype=theano.config.floatX

class ComputeConfidenceInterval(object):
    #
    def __init__(self, settings):
        self.path_to_results = settings['path_to_results']
        self.confidence_level = settings[
            'confidence_level'
        ]
        self.num_bootstrap = settings[
            'num_bootstrap'
        ]
        self.predict_lambda = settings[
            'predict_lambda'
        ]
        self.quantile_low = 0.5 * (
            1.0 - self.confidence_level
        ) * 100.0
        self.quantile_high = self.quantile_low + self.confidence_level * 100.0
    #
    def set_params(self, settings):
        self.path_to_results = settings['path_to_results']
        self.confidence_level = settings[
            'confidence_level'
        ]
        self.num_bootstrap = settings[
            'num_bootstrap'
        ]
        self.quantile_low = 0.5 * (
            1.0 - self.confidence_level
        ) * 100.0
        self.quantile_high = self.quantile_low + self.confidence_level * 100.0
        #
    #
    def read_results(self):
        with open(self.path_to_results, 'rb') as f:
            self.dict_results = pickle.load(f)
        #

    #
    #
    def bootstrap(self):
        if self.predict_lambda:
            self.bootstrap_intensity()
        else:
            self.bootstrap_likelihood()
    #
    #
    def bootstrap_likelihood(self):
        #
        if self.path_to_results != None:
            self.read_results()
        #
        array_log_likelihood_seq = numpy.array(
            self.dict_results['lists']['log_likelihood_seq']
        )
        array_log_likelihood_type = numpy.array(
            self.dict_results['lists']['log_likelihood_type']
        )
        array_log_likelihood_time = numpy.array(
            self.dict_results['lists']['log_likelihood_time']
        )
        array_num_of_events = numpy.array(
            self.dict_results['lists']['num_of_events']
        )
        #
        num_seqs = len(
            self.dict_results['lists']['log_likelihood_seq']
        )
        #
        list_log_likelihood_seq = []
        list_log_likelihood_type = []
        list_log_likelihood_time = []
        #
        for idx_bootstrap in range(self.num_bootstrap):
            list_idx_selected = numpy.random.choice(
                a = num_seqs, size = num_seqs,
                replace = True
            )
            list_log_likelihood_seq.append(
                numpy.sum(
                    array_log_likelihood_seq[list_idx_selected]
                ) / numpy.sum(
                    array_num_of_events[list_idx_selected]
                )
            )
            list_log_likelihood_type.append(
                numpy.sum(
                    array_log_likelihood_type[list_idx_selected]
                ) / numpy.sum(
                    array_num_of_events[list_idx_selected]
                )
            )
            list_log_likelihood_time.append(
                numpy.sum(
                    array_log_likelihood_time[list_idx_selected]
                ) / numpy.sum(
                    array_num_of_events[list_idx_selected]
                )
            )
            if idx_bootstrap % 100 == 99 :
                print "in bootstrapping, the step is out of ", idx_bootstrap, self.num_bootstrap
        list_log_likelihood_seq.sort()
        list_log_likelihood_type.sort()
        list_log_likelihood_time.sort()
        #
        list_log_likelihood_seq = numpy.array(
            list_log_likelihood_seq
        )
        list_log_likelihood_type = numpy.array(
            list_log_likelihood_type
        )
        list_log_likelihood_time = numpy.array(
            list_log_likelihood_time
        )
        #
        mean_seq = round(
            numpy.mean(list_log_likelihood_seq), 4
        )
        mean_type = round(
            numpy.mean(list_log_likelihood_type), 4
        )
        mean_time = round(
            numpy.mean(list_log_likelihood_time), 4
        )
        #
        low_seq = numpy.percentile(
            a = list_log_likelihood_seq, q = self.quantile_low
        )
        low_seq = round(low_seq, 4)
        high_seq = numpy.percentile(
            a = list_log_likelihood_seq, q = self.quantile_high
        )
        high_seq = round(high_seq, 4)
        #
        low_type = numpy.percentile(
            a = list_log_likelihood_type, q = self.quantile_low
        )
        low_type = round(low_type, 4)
        high_type = numpy.percentile(
            a = list_log_likelihood_type, q = self.quantile_high
        )
        high_type = round(high_type, 4)
        #
        low_time = numpy.percentile(
            a = list_log_likelihood_time, q = self.quantile_low
        )
        low_time = round(low_time, 4)
        high_time = numpy.percentile(
            a = list_log_likelihood_time, q = self.quantile_high
        )
        high_time = round(high_time, 4)
        #
        print "First print original results for sanity-check : "
        print self.dict_results['log_likelihood']
        #
        print "Here is the results of bootstrapping : "
        print "the confidence level is : ", self.confidence_level
        print "for the seq, i.e., both type and time : "
        print "the mean is : ", mean_seq
        print "the interval is : (" + str(low_seq) + ',' + str(high_seq) + ') '
        print "the combo is : (" + str(low_seq) + ',' + str(mean_seq) + ',' + str(high_seq) + ')'
        print "for the type : "
        print "the mean is : ", mean_type
        print "the interval is : (" + str(low_type) + ',' + str(high_type) + ') '
        print "the combo is : (" + str(low_type) + ',' + str(mean_type) + ',' + str(high_type) + ') '
        print "for the time : "
        print "the mean is : ", mean_time
        print "the interval is : (" + str(low_time) + ',' + str(high_time) + ') '
        print "the combo is : (" + str(low_time) + ',' + str(mean_time) + ',' + str(high_time) + ') '
        #
        #TODO: save the boostrap results into results dict
        dict_bootstrap = {
            'confidence_level': self.confidence_level,
            'seq': {
                'low': low_seq, 'high': high_seq
            },
            'type': {
                'low': low_type, 'high': high_type
            },
            'time': {
                'low': low_time, 'high': high_time
            }
        }
        self.dict_results['bootstrap'] = dict_bootstrap
        #
        print "saving into result pickle ... "
        with open(self.path_to_results, 'wb') as f:
            pickle.dump(self.dict_results, f)
        #
        print "finished bootstrapping "
        #
        #
    #
    #
    #
    def bootstrap_intensity(self):
        # compute bootstrap interval for intensity
        # for intensity, lists has tensors as items
        # value : (intensity-realintensity)^2
        # shape : K * N * 1
        if self.path_to_results != None:
            self.read_results()
        #
        list_lambda = self.dict_results['lists']['lambda_samples']
        list_lambda_gold = self.dict_results['lists']['lambda_samples_gold']
        list_num_of_samples = self.dict_results['lists']['num_of_samples']
        #
        num_seqs = len(list_lambda)
        list_ratio = [] # save boostrap results
        list_ratio_2 = []
        # _2 is the variance of real intensity
        # scaled by square of avg real intensity
        list_ratio_total = []
        list_ratio_2_total = []
        # to compare the total intensity
        #
        for idx_bootstrap in range(self.num_bootstrap):
            list_idx_selected = numpy.random.choice(
                a=num_seqs, size=num_seqs, replace=True
            )
            total_up, total_average = 0.0, 0.0
            total_num = 0.0
            total_down = 0.0
            total_downdown = 0.0
            #
            total_up_total = 0.0
            total_average_total = 0.0
            total_down_total = 0.0
            total_downdown_total = 0.0
            # these for total intensity
            #
            for idx_selected in list_idx_selected:
                # each item in list is a tensor
                # K * N * 1
                lambda_samples = list_lambda[idx_selected]
                lambda_samples_gold = list_lambda_gold[idx_selected]
                num_of_samples = list_num_of_samples[idx_selected]
                #
                total_up += numpy.sum(
                    (
                        lambda_samples-lambda_samples_gold
                    )**2,
                    axis=(1,2)
                )
                total_average += numpy.sum(
                    lambda_samples_gold, axis=(1,2)
                )
                total_num += numpy.float32(num_of_samples)
                #
                total_up_total +=numpy.sum(
                    (
                        numpy.sum(lambda_samples, axis=0) - numpy.sum(lambda_samples_gold,axis=0)
                    )**2
                )
                total_average_total += numpy.sum(
                    lambda_samples_gold
                )
                #
            #
            average_intensity = total_average / total_num
            average_intensity_total = total_average_total / total_num
            #
            for idx_selected in list_idx_selected:
                lambda_samples_gold = list_lambda_gold[idx_selected]
                #
                total_down += numpy.sum(
                    (
                        lambda_samples_gold-average_intensity[:,None,None]
                    )**2,
                    axis=(1,2)
                )
                #
                total_downdown += numpy.sum(
                    (
                        0.0*lambda_samples_gold-average_intensity[:,None,None]
                    )**2,
                    axis=(1,2)
                )
                #
                total_down_total += numpy.sum(
                    (
                        numpy.sum(lambda_samples_gold,axis=0)-average_intensity_total
                    )**2
                )
                total_downdown_total += numpy.sum(
                    (
                        0.0*numpy.sum(lambda_samples_gold,axis=0)-average_intensity_total
                    )**2
                )
                #
            ratio = total_up / total_down
            ratio_2 = total_down / total_downdown
            # K *
            ratio_total = total_up_total / total_down_total
            ratio_2_total = total_down_total / total_downdown_total
            #
            list_ratio.append(numpy.copy(ratio) )
            list_ratio_2.append(numpy.copy(ratio_2) )
            #
            list_ratio_total.append(numpy.copy(ratio_total))
            list_ratio_2_total.append(numpy.copy(ratio_2_total))
            #
        ratios = numpy.array(list_ratio)
        ratios_2 = numpy.array(list_ratio_2)
        # M * K
        mean_ratios = numpy.mean(ratios, axis=0)
        mean_ratios_2 = numpy.mean(ratios_2, axis=0)
        #
        # total intensity
        ratios_total = numpy.array(list_ratio_total)
        ratios_2_total = numpy.array(list_ratio_2_total)
        mean_ratios_total = numpy.mean(ratios_total)
        mean_ratios_2_total = numpy.mean(ratios_2_total)
        #
        #
        dict_bootstrap = {
            'confidence_level': self.confidence_level,
            'mean': mean_ratios,
            'mean_2': mean_ratios_2
        }
        #
        print "for this trial : "
        print "model is : ", self.dict_results['model']
        print "model gold is : ", self.dict_results['model_gold']
        print "for total intensity : "
        print "MSE/VAR and VAR/AVG^2 are : ", mean_ratios_total, mean_ratios_2_total
        #TODO: predict individual results
        #print "for all types , means are : ", mean_ratios
        #print "real variance/square_mean : ", mean_ratios_2
        #
        mean_mean = 0.0
        #
        for k in range(ratios.shape[1]):
            ratios_k = ratios[:,k]
            low = numpy.percentile(
                a=ratios_k,
                q=self.quantile_low
            )
            high = numpy.percentile(
                a=ratios_k,
                q=self.quantile_high
            )
            low = round(low, 6)
            high = round(high, 6)
            #TODO: predict individual results
            #print "for type : ", str(k+1)
            #print "C.I. is : ("+str(low)+','+str(round(mean_ratios[k],6))+','+str(high)+')'
            dict_bootstrap['type_'+str(k+1)] = {
                'low': low, 'mean': mean_ratios[k], 'high': high
            }
            mean_mean +=  mean_ratios[k]
        mean_mean /= 1.0*ratios.shape[1]
        #
        print "Mean over K types : "
        print "MSE/Var : ", round(numpy.mean(mean_ratios), 6)
        print "Var/AVG^2 : ", round(numpy.mean(mean_ratios_2), 6)
        #
        print "saving into result pickle ... "
        with open(self.path_to_results, 'wb') as f:
            pickle.dump(self.dict_results, f)
        print "finish bootstrapping for intensity eval"
    #
    #
    #


class PermutationTest(object):
    #
    def __init__(self, settings):
        self.path_to_results = settings['path_to_results']
        #
        self.path_0 = self.path_to_results[0]
        self.path_1 = self.path_to_results[1]
        #
        self.num_permuation = settings[
            'num_permuation'
        ]
    #
    def set_params(self, settings):
        self.path_to_results = settings['path_to_results']
        #
        self.path_0 = self.path_to_results[0]
        self.path_1 = self.path_to_results[1]
        #
        self.num_permuation = settings[
            'num_permuation'
        ]
        #
    #
    def read_results(self):
        with open(self.path_0, 'rb') as f:
            self.dict_results_0 = pickle.load(f)
        with open(self.path_1, 'rb') as f:
            self.dict_results_1 = pickle.load(f)
        #
    #
    #
    def compute_p(self):
        # assume there are array_0 and array_1
        # This is my version ...
        len_array = len(self.array_0)
        diff_array = self.array_0 - self.array_1
        threshold = numpy.abs(numpy.mean(diff_array))
        #
        cnt = 0.0
        #
        for idx_perm in range(self.num_permuation):
            #'''
            list_idx_selected = numpy.random.choice(
                [True, False], size = (len_array, ),
                replace = True
            )
            diff_array_temp = numpy.copy(diff_array)
            diff_array_temp[
                list_idx_selected
            ] = -1.0 * numpy.copy(
                diff_array[list_idx_selected]
            )
            shot_new = numpy.abs(
                numpy.mean(diff_array_temp)
            )
            #print "threshold is : ", threshold
            #print "shot is : ", shot_new
            if shot_new >= threshold:
                cnt += 1.0
            #'''
        print "cnt is : ", cnt
        value_p = 1.0 * cnt / self.num_permuation
        print "p value is : ", value_p
        return value_p
    #
    def compute_p_2(self):
        # assume there are array_0 and array_1
        # This is very likely to be wrong !!!!
        # Treat it as deprecated !!!
        #
        len_array = len(self.array_0)
        diff_array = self.array_0 - self.array_1
        threshold = numpy.abs(numpy.mean(diff_array))
        #
        '''
        to_shuffle = numpy.concatenate(
            [self.array_0, self.array_1]
        )
        '''
        #
        cnt = 0.0
        #
        for idx_perm in range(self.num_permuation):
            idx_pos = idx_perm % len_array
            diff_array[idx_pos] = -1.0 * diff_array[idx_pos]
            shot_new = numpy.abs(numpy.mean(diff_array))
            if shot_new >= threshold:
                cnt += 1.0
        print "cnt is : ", cnt
        value_p = 1.0 * cnt / self.num_permuation
        print "p value is : ", value_p
        return value_p
    #
    def save_csv(self):
        self.read_results()
        l_0 = self.dict_results_0['lists']['log_likelihood_time']
        l_1 = self.dict_results_1['lists']['log_likelihood_time']
        names_field = ['m0', 'm1']
        with open('./sample_results.csv', 'w') as f:
            writer_csv = csv.DictWriter(
                f, fieldnames = names_field
            )
            writer_csv.writeheader()
            for i_0, i_1 in zip(l_0, l_1):
                dict_temp = {
                    'm0': round(i_0, 4),
                    'm1': round(i_1, 4)
                }
                #print dict_temp
                writer_csv.writerow(
                    dict_temp
                )
        print "done ! "
    #
    #
    def pair_perm_test_DW(self, x, y, num_samples=99999):
        # Code borrowed from Dingquan Wang
        x, y = map(numpy.array, [x, y])
        d = x - y
        #print d
        dim = len(d)
        m0 = numpy.mean(d)
        permutation = (
            numpy.random.binomial(
                1, .5, (num_samples, dim)
            ) * 2 - 1
        ) * d
        mean_perm = numpy.mean(permutation, 1)
        return float(sum(abs(mean_perm) >= abs(m0))) / mean_perm.shape[0]
    #
    #
    def permutation_test(self):
        #
        self.read_results()
        #
        for_loop = [
            'log_likelihood_seq',
            'log_likelihood_type',
            'log_likelihood_time'
        ]
        #
        dict_p = {}
        #
        for tag_current in for_loop:
            '''
            self.array_0 = numpy.copy(
                self.dict_results_0['lists'][tag_current]
            )
            self.array_1 = numpy.copy(
                self.dict_results_1['lists'][tag_current]
            )
            '''
            #
            self.array_0 = numpy.copy(
                numpy.array(
                    self.dict_results_0['lists'][tag_current]
                ) / numpy.array(
                    self.dict_results_0['lists'][
                        'num_of_events'
                    ]
                )
            )
            self.array_1 = numpy.copy(
                numpy.array(
                    self.dict_results_1['lists'][tag_current]
                ) / numpy.array(
                    self.dict_results_1['lists'][
                        'num_of_events'
                    ]
                )
            )
            #
            # test out of curiosity
            # need to be commented out in the usage
            #self.array_1 = self.array_1 + 0.8*numpy.mean(
            #    self.array_0 - self.array_1
            #)
            #
            #dict_p[tag_current] = self.compute_p()
            dict_p[tag_current] = round(
                self.pair_perm_test_DW(
                    self.array_0, self.array_1,
                    self.num_permuation
                ), 4
            )
            #
            diff_vec = self.array_0 - self.array_1
            diff_mean = numpy.mean(diff_vec)
            diff_std = numpy.std(diff_vec)
            diff_max = numpy.max(diff_vec)
            diff_min = numpy.min(diff_vec)
            #
            print "For this tag : ", tag_current
            #print "Here is some statistics about the diff : "
            #print "mean is : ", diff_mean
            #print "std is : ", diff_std
            #print "max is : ", diff_max
            #print "min is : ", diff_min
            #print "How many 0 >= 1 : "
            #print sum(self.array_0 >= self.array_1 )
            #print "How many 0 < 1 : "
            #print sum(self.array_1 >= self.array_0 )
            #print "mean and std for 0 is : ", numpy.mean(self.array_0), numpy.std(self.array_0)
            #print "mean and std for 1 is : ", numpy.mean(self.array_1), numpy.std(self.array_1)
            print "the p value for this tag is : "
            #print tag_current
            print str(dict_p[tag_current])
        #
        print "The values of P in order of seq/type/time : "
        print str(dict_p['log_likelihood_seq']) + ' / ' + str(dict_p['log_likelihood_type']) + ' / ' + str(dict_p['log_likelihood_time'])
        #
        print "Finished permuation test "
        #
        #
        '''
        print "total, seq, type, time : "
        print self.num_permuation, cnt_seq, cnt_type, cnt_time
        #
        p_seq = round(cnt_seq/(1.0*self.num_permuation), 4)
        p_type = round(cnt_type/(1.0*self.num_permuation), 4)
        p_time = round(cnt_time/(1.0*self.num_permuation), 4)
        #
        #
        print "How many 0 >= 1 : "
        print sum(array_log_likelihood_seq_0 >= array_log_likelihood_seq_1)
        print "How many 0 < 1 : "
        print sum(array_log_likelihood_seq_0 < array_log_likelihood_seq_1)
        #
        #
        print "Here is the results of permutation test : "
        print "The two results files are : "
        print self.path_0
        print self.path_1
        print "The values of P in order of seq / type / time : "
        print str(p_seq) + ' / ' + str(p_type) + ' / ' + str(p_time)
        #
        print "Finished permuation test "
        #
        '''
        #

    #
