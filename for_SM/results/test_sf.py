#!/bin/python
#
#This code for validate symmetry function as pytorch save file
#Need to do 
#Convert atomic position to symmetry function 
#Check sf value in torch & pickle and validate
#
import os, sys
import subprocess as sbp
import ase
from ase import io
from ase import units
import torch
import pickle
import six


#load pickle file(tempoaray) at util.__init__
def pickle_load(filename):
    with open(filename, 'rb') as fil:
        if six.PY2:
            return pickle.load(fil)
        elif six.PY3:
            return pickle.load(fil, encoding='latin1')
                                                 
#read parameters from file at symmetry_function.__init__
def read_params(filename):
    params_i = list()
    params_d = list()
    with open(filename, 'r') as fil:
        for line in fil:
        tmp = line.split()
        params_i += [list(map(int,   tmp[:3]))]
        params_d += [list(map(float, tmp[3:]))]                
        params_i = np.asarray(params_i, dtype=np.intc, order='C')
        params_d = np.asarray(params_d, dtype=np.float64, order='C')  
    return params_i, params_d

#testing temporary code
if __name__ == '__main__':
    data1 = pickle_load('data1.pickle')
    print(data1.keys())
    pass    
