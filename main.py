#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:55:44 2021

@author: staszek
"""
import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
import copy


data = np.genfromtxt('test_data/OCP16 Ntag 2mm 470nm 0.4uJ.csv', delimiter=',')


t_vect = data[0,1:]
a_vect = data[50,1:] #at 500nm good for tests

error = np.std(a_vect[np.where(t_vect <= -0.5)])

t_vect = t_vect[20:]
a_vect = a_vect[20:]


def gaussExp(t, sigma, A, tau):
    eksp = np.exp(-(t - sigma**2 / (2 * tau))/tau)/2
    erfu = special.erf((t - sigma**2/tau)/(np.sqrt(2)*sigma)) + 1
    return A * eksp * erfu

def buildExpArray(t_vect, tauspace, t0, sigma):
    tmp = np.zeros([t_vect.shape[0], tauspace.shape[0]])
    
    for i in range(tauspace.shape[0]):
        tmp[:,i] = gaussExp(t_vect-t0, sigma, 1.0, tauspace[i])
    return tmp


tauspace = np.logspace(-1.0, 3.0, num=50) #tau values used to build the distribution
tauamps = tauspace*0.0
alpha = 1 #hyper parameter of S contribution to chi2,

exparray = buildExpArray(t_vect, tauspace, 0.0, 0.0001)



def calculateKinetic(p):
    a_model_vect = exparray.dot(p)
    return a_model_vect

def calcEntropy(p):
    entvect = np.abs(p) - np.abs(p)*np.log(np.abs(p)) #Skilling
    for i in range(entvect.shape[0]): #bad solution
        if(np.isnan(entvect[i])):
            entvect[i] = 0.0
    return np.sum(entvect)
    #problems:
    #minor: amps over 1 will give negative entropy
    #major: amps with zero will give nan


def residualWithRegularisation(p):
    values = calculateKinetic(p)
    residuals = (values - a_vect)/(error*values.shape[0])
    entropy = calcEntropy(p)
    chi2 = np.sum(np.power(residuals,2))
    print("Entropy is: " + str(entropy) + ", and chi2 is: " + str(chi2))
    return chi2 + alpha*entropy #should be with minus... it seems that minimizing etropy gives something, instead of maximizing. makes no sense at this point...

def plotKinetic(p):
    plt.figure()
    plt.plot(t_vect, a_vect, "bo")
    plt.plot(t_vect, calculateKinetic(p), "r-")
    plt.xscale(value = "log")        
    plt.show()        
    
def plotDistribution(p):
    plt.figure()
    plt.plot(tauspace, p, "bo")  
    plt.xscale(value = "log")
    plt.show() 


p = copy.deepcopy(tauspace)
for i in range(tauspace.shape[0]):
    p[i] = 0.00000001

out_params = optimize.minimize(residualWithRegularisation, p)

print(out_params)

plotKinetic(out_params.x)
plotDistribution(out_params.x)
