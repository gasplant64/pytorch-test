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
import yaml
import numpy as np
import math
from math import pi



##### File processing #####
#load pickle file(tempoaray) at util.__init__
def pickle_load(filename):
    with open(filename, 'rb') as f:
        if six.PY2:
            return pickle.load(f)
        elif six.PY3:
            return pickle.load(f, encoding='latin1')
                                                 
#read parameters from file at symmetry_function.__init__
# Parameter formats
# [type of SF(1)] [atom type index(2)] [cutoff distance(1)] [coefficients for SF(3)]
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

#Get parameters file directory
def get_params_dir(filename):
    params_dir = list()
    input_yaml = read_yaml(filename)
    atom_type = input_yaml['atom_types']
    for atom in atom_type:
        params_dir.append(input_yaml['symmetry_function']['params'][atom])
    return atom_type , params_dir

#Read VASP-OUTCAR file (only applicable VASP-OUTCAR format) 
#At Symmetry_function.generate in symmetry_function.__init__
def open_outcar(filename):
    if ase.__version__ >= '3.18.0':
        snapshot = io.read(filename, index=-1, format='vasp-out')
    else:
        snapshot = io.read(filename, index=-1, format='vasp-out', force_consistent=True)
    return snapshot

#Read yaml file 
def read_yaml(filename):
    with open(filename) as f:
        output = yaml.safe_load(f)
    return output

##### Symmetry function processing #####
#Run symmetry functions as input ase.atoms
def cutoff_func(dist , cutoff):
    out = math.cos(pi*dist/cutoff) + 1.0
    out = 0.5*float(out)
    return out

#G2 symmetry functions 
def g2_ker(dist ,cutoff,eta , r_s = 0):
    out = math.exp(-eta*(dist-r_s)**2)*cutoff_func(dist , cutoff)
    return out

#G4 symmetry functions 
def g4_ker(dist_ij,dist_ik,dist_jk, angle ,cutoff,eta ,zeta ,lamda):
    out  = g2_ker(dist_ij, cutoff , eta)*g2_ker(dist_ik, cutoff , eta)*g2_ker(dist_jk, cutoff , eta)
    out *= (1 + lamda * math.cos(pi/180.0*angle))**zeta
    return out

#G5 symmetry functions 
def g5_ker(dist_ij,dist_ik, angle ,cutoff,eta ,zeta ,lamda):
    out  = g2_ker(dist_ij, cutoff , eta)*g2_ker(dist_ik, cutoff , eta)
    out *= (1 + lamda * math.cos(pi/180.0*angle))**zeta
    return out

#Decorator function / preprocess distance & angle of atomic environemnt needed 
#index -> SF type & param_d -> parameters of SF
def generate_sf(index , param_d):
    cutoff = param_d[0]
    if index == 2: 
        eta = param_d[1]; r_s = param_d[2]
        # G2 SF 
        def sym_func(dist_list):
            output = 0
            for dist in dist_list:
                output += g2_ker(dist,cutoff,eta)
            return output
    elif index == 4: 
        eta = param_d[1]; zeta = param_d[2] ; lamda = param_d[2]
        # G4 SF 
        def sym_func(dist_list):
            output = 0
            angle = 0
            # dist = [dist_ij,dist_ik ,dist_jk] type & angle generated
            for dist in dist_list:
                angle = 180/pi*math.acos((dist[0]**2+dist[1]**2-dist[2]**2)/(2*dist[0]*dist[1]))
                output += g4_ker(dist[0],dist[1],dist[2],angle,cutoff,eta ,zeta ,lamda)
            output *= 2.0**(1-zeta)
            return output
    elif index == 5: 
        eta = param_d[1]; zeta = param_d[2] ; lamda = param_d[2]
        # G4 SF 
        def sym_func(dist_list):
            output = 0
            # dist = [dist_ij,dist_ik, dist_jk] type
            for dist in dist_list:
                angle = 180/pi*math.acos((dist[0]**2+dist[1]**2-dist[2]**2)/(2*dist[0]*dist[1]))
                output += g5_ker(dist[0],dist[1],angle,cutoff,eta ,zeta ,lamda)
            output *= 2.0**(1-zeta)
            return output
    return sym_func


#Processing OUTCAR



#Class that generate list of symmetry function 
class Test_sf:
    def __init__(self , atoms):
        self.atoms = atoms # ase.atoms
        self.atom_type = None
        self.list_sf = list()
    
    #Calculating sf
    def calculate(self):
        out = None
        return out



#testing temporary code
if __name__ == '__main__':
    #Get params directroy from yaml file
    #_ , params_dir = get_params_dir('..\\input.yaml')
    #print(params_dir)


    #Check read parameters
    #par_i , par_d = read_params('..\\params_Ge')
    #print(par_i)
    #print(par_d)

    #Check open OUTCAR(VASP)
    #outcar = open_outcar('OUTCAR_1')
    #print(outcar)


    # #Check SFs
    # print('CUTOFF FUNCTION')
    # print(cutoff_func(dist = 4 ,cutoff= 5)) #CUTOFF function
    # #Radial function
    # print('G2 FUNCTION')
    # print(g2_ker(dist = 2 ,cutoff= 11.3,eta = 3 , r_s=3)) # G2 function
    # #Angular functions
    # print('ANGULAR FUNCTION')
    # print(g4_ker(dist_ij= 0 ,dist_ik= 0,dist_jk= 0, angle= 30 ,cutoff= 10,eta = 0.1 , zeta=1 , lamda = 1))
    # print(g4_ker(dist_ij= 0 ,dist_ik= 0,dist_jk= 0, angle= 30 ,cutoff= 10,eta = 0.1 , zeta=1 , lamda = -1))
    # print(g4_ker(dist_ij= 0 ,dist_ik= 0,dist_jk= 0, angle= 30 ,cutoff= 10,eta = 0.1 , zeta=64 , lamda = 1)*(2**-63))
    # #Radial part in angulart function
    # print('G4 & G5 FUNCTION')
    # print(g4_ker(dist_ij= 2 ,dist_ik= 2,dist_jk= 2, angle= 60 ,cutoff= 12,eta = 0.1 , zeta=1 , lamda = 0))
    # print(g5_ker(dist_ij= 2 ,dist_ik= 2, angle= 60 ,cutoff= 12,eta = 0.1 , zeta=1 , lamda = 0))
    # ##All checked

    #Testing generate SF
    test_g2 = generate_sf(index = 2,param_d=[6.0,0.003214 , 0.0 , 0.0])
    print('Testing G2:  ',test_g2([2.0,1.0,3.0]))
    test_g4 = generate_sf(index = 4,param_d=[6.0,0.089277, 1.0, -1.0])
    test_g5 = generate_sf(index = 5,param_d=[6.0,0.089277, 1.0, -1.0])
    print('Testing G4:  ',test_g4([[2.0,1.0,1.5],[2.0,1.0,2.5]]))
    print('Testing G5:  ',test_g5([[2.0,1.0,1.5],[2.0,1.0,2.5]]))