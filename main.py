#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:55:44 2021

@author: staszek
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import lmfit
import copy


data = np.genfromtxt('test_data/OCP16 Ntag 2mm 470nm 0.4uJ.csv', delimiter=',')


t_vect = data[0,1:]
a_vect = data[50,1:] #at 500nm good for tests

t_vect = t_vect[20:]
a_vect = a_vect[20:]

def gaussExp(t, sigma, A, tau):
    eksp = np.exp(-(t - sigma**2 / (2 * tau))/tau)/2
    erfu = sp.special.erf((t - sigma**2/tau)/(np.sqrt(2)*sigma)) + 1
    return A * eksp * erfu

def buildExpArray(t_vect, tauspace, t0, sigma):
    tmp = np.zeros([t_vect.shape[0], tauspace.shape[0]])
    
    for i in range(tauspace.shape[0]):
        tmp[:,i] = gaussExp(t_vect-t0, sigma, 1.0, tauspace[i])
    return tmp


errorsq = np.power(np.std(a_vect[np.where(t_vect <= -0.5)]),2)

tauspace = np.logspace(-1.0, 3.0, num=10) #tau values used to build the distribution
tauamps = tauspace*0.0
alpha = 1 #hyper parameter of S contribution to chi2,

exparray = buildExpArray(t_vect, tauspace, 0.0, 0.001)

def residual(params):
    p = np.array([x.value for x in params.values()])
    
    a_model_vect = exparray.dot(p)

    return a_model_vect - a_vect



p = lmfit.Parameters()
for i in range(tauspace.shape[0]):
    p.add("A"+str(i), 0.00000001)
    
if(tauspace.shape[0] >= t_vect.shape[0]):
    raise Exception("Space of taus has to be smaller that number of delays!")

mini = lmfit.Minimizer(residual, p, nan_policy='propagate')
out = mini.minimize(method='leastsq')
out_params = out.params
lmfit.report_fit(out.params)


#PLAN:
#1. load kinetic
#2. opt constant line of amps
#3. start optimization with given alpha
#4.  ....
