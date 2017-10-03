# -*- coding: utf-8 -*-
"""

Here are the utility modules

@author: hongyuan
"""

import numpy
import pickle
import os
import json
import time
from operator import itemgetter

import theano
from theano import sandbox
import theano.tensor as tensor

dtype = theano.config.floatX

def sample_weights(numrow, numcol):
    values = numpy.ndarray([numrow, numcol], dtype=dtype)
    for coli in xrange(numcol):
        vals = numpy.random.uniform(low=-1., high=1.,  size=(numrow,))
        #vals_norm = numpy.sqrt((vals**2).sum())
        #vals = vals / vals_norm
        values[:,coli] = vals
    _,svs,_ = numpy.linalg.svd(values)
    #svs[0] is the largest singular value
    values = values / svs[0]
    return values
