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
    for i1 in range(np.shape(intermat)[0]): #Source
        for i2 in range(np.shape(intermat)[1]): #Target
            if intermat[i1,i2] == 0.:
                term[i1,i2] = 1
            elif intermat[i1,i2] == 1.:
                term[i1,i2] = exp[2*i1+1]**hill/(1+exp[2*i1+1]**hill)
            elif intermat[i1,i2] == 2.:
                term[i1,i2] = 1/(1+exp[2*i1+1]**hill)
    return np.prod(term,axis=0)

def diff_eq(t,exp,intermat,alpha,beta,basal,
            hill,const=False,sig=1.0,name='sig'):
    #intermat = intermat.T[nodes!=sig].T[nodes!=sig]
    nodes = np.array(intermat.columns)
    if const == True: 
        exp = np.append(exp,[sig,sig])
    dydt = np.ones(len(nodes)*2)
    term = hill_fn(np.array(intermat),nodes,exp,hill)
    for n,y in enumerate(nodes):
        #Protein equation
        dydt[2*n+1] = -beta[n]*(exp[2*n+1]-exp[2*n])
        #mRNA equation
        dydt[2*n] = -exp[2*n] + alpha[n]*term[n]
    return dydt
    
#@jit(nopython=True)
def peak_finder(bool_peaks):
    res = np.zeros(np.shape(bool_peaks),dtype=bool)
    res = np.append(res,False)
    peak = 0;
    for n,val in enumerate(bool_peaks[:-1]):
        if val == True and bool_peaks[n+1] == False: 
            res[n-int(peak/2)] = True
        elif val == True: peak = peak+1
        else: peak = 0
    return res[:-1]
            
#@jit(nopython=True)
def slid_avg(times):
    diff = []
    for i in range(len(times)-1):
        diff.append(times[i+1]-times[i])
    return np.mean(diff)

def amp_freq(data,time):
    #plt.plot(time,data)
    nlen = len(data)
    data = data[int(nlen*0.2):]
    time = time[int(nlen*0.2):]
    maxi = np.max(data)
    mini = np.min(data)
    higher = data > 0.95*maxi
    peaks = peak_finder(higher)
    peak_times = time[peaks]
    peak_vals = data[peaks]
    avg_amp = np.mean(peak_vals)
    avg_time = slid_avg(peak_times)
    return(avg_amp,avg_time)
    
def evol_reader(base,nodes,signal):
    tot = 3**np.shape(base)[1]
    res = []
    for i in range(tot):
        mat = np.copy(base)
        app = np.array(list(np.base_repr(i,base=3)
                            .zfill(np.shape(mat)[1]))
                       ).astype(np.float64)
        mat = np.append(mat,[app],axis=0)
        res.append(mat)
    return np.array(res),np.append(nodes,signal)
    