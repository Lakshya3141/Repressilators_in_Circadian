"""
Functions to parse network topology and input conditions
@author: Lakshya Chauhan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time,random
from numba import jit

def adjacency(in_file):
    fn = os.getcwd()+'\\input\\'+in_file
    node_data = pd.read_table(fn+'.ids')
    nodes = list(node_data['Node'])
    num = len(nodes)
    intermat = pd.DataFrame(np.zeros([num,num]),
                            columns=nodes,index=nodes)
    topo = open(fn+'.topo', "r").readlines()[1:]
    for line in topo:
        s = line.split('\t')[0]
        t = line.split('\t')[1]
        k = line.split('\t')[2].strip('\n')
        intermat[t][s] = k
    return np.array(nodes),intermat,np.array(node_data['alpha']),np.array(node_data['beta']),np.array(node_data['basal'])

def inhibition(intermat):
    return np.array(intermat==2).astype(float)

def activation(intermat):
    return np.array(intermat==1).astype(float)

#@jit(nopython=True)
def hill_fn(intermat,nodes,exp,hill=2):
    term = intermat.copy()
    for i1,n1 in enumerate(nodes): #Source
        for i2,n2 in enumerate(nodes): #Target
            if intermat[i1,i2] == 0.:
                term[i1,i2] = 1
            elif intermat[i1,i2] == 1.:
                term[i1,i2] = exp[2*i1+1]**hill/(1+exp[2*i1+1]**hill)
            elif intermat[i1,i2] == 2.:
                term[i1,i2] = 1/(1+exp[2*i1+1]**hill)
    return np.prod(term,axis=0)

def diff_eq(t,exp,intermat,alpha,beta,basal,hill,const=False,sig=1.0):
    #intermat = intermat.T[nodes!=sig].T[nodes!=sig]
    nodes = np.array(intermat.columns)
    if const == False:
        dydt = np.ones(len(nodes)*2)*sig
        term = hill_fn(np.array(intermat),nodes,exp,hill)
        for n,y in enumerate(nodes):
            #Protein equation
            dydt[2*n+1] = -beta[n]*(exp[2*n+1]-exp[2*n])
            #mRNA equation
            dydt[2*n] = -exp[2*n] + alpha[n]*term[n]
    return dydt
    