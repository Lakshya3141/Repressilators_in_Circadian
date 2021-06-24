"""
Code to produce, simulate and store data for single signal perturbed networks
Can also analyse already generated data
Change boolean holders accordingly
@author: Lakshya Chauhan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time,random
import util_funcs as uf
global nodes, intermat
import scipy.integrate as scip
from scipy.fftpack import fft, fftfreq
import pickle as pk

# if not os.path.exists('Output'):
#     os.makedirs('Output')
    
global hill
hill = 2

gen = False
read = True
img = False
ana = True
img_freq = True

in_file = 'repD'

# if not os.path.exists('./Output/{}'.format(in_file)):
#     os.makedirs('./Output/{}'.format(in_file))
    
nodes,intermat,alpha,beta,basal = uf.adjacency(in_file)
init = np.random.rand(len(nodes)*2)*10
#init = np.array([0,0,900,900])

obs_node = 'C'
ind = np.where(nodes == obs_node)[0][0]

t_fin = 100
dt_max = 0.01
steps = int(t_fin/dt_max)
times = np.linspace(0,t_fin,steps)
evo,source = uf.evol_reader(np.array(intermat),nodes,'sig')

exp_sig = np.linspace(10,2*max(alpha),100)
cnt = 0

if gen == True:
    solutions = []
    for i,n in enumerate(evo):
        solutions.append([])
        series = []
        for e in exp_sig:
            mat = pd.DataFrame(evo[i],columns=nodes,index=source)
            sol = scip.solve_ivp(uf.diff_eq, (0,t_fin), y0 = init, t_eval=times,
                           args=(mat,alpha,beta,basal,hill,True,e))
            solutions[i].append(sol.y[2*ind+1])
            series.append(sol.y[2*ind+1])
            cnt = cnt+1
            print("Done: ",cnt," out of ",len(evo)*len(exp_sig))
            
        df = pd.DataFrame(series, columns = times,index = exp_sig).T
        path = 'Output/{}/pert{}'.format(in_file,i)
        if not os.path.exists(path):
            os.makedirs(path)
        df.to_excel('{}/data.xlsx'.format(path))
    pk.dump(solutions, open('./Output/{}/bin.data'.format(in_file), 'wb'))
        
if read == True:
    solutions = pk.load(open('./Output/{}/bin.data'.format(in_file), 'rb'))

if img == True:
    for e in [2,30,50,60,98]:
    #for e in [2,30]:
        fig, axs = plt.subplots(3, 9)
        fig_ = plt.gcf()
        fig_.set_size_inches(30, 17.5)
        for i in range(len(evo)):
            axs[int(i/9), i%9].plot(times, solutions[i][e])
            axs[int(i/9), i%9].set_title(i)
        fig.savefig('Output/{}/perts_{}'.format(in_file,int(exp_sig[e])))
        plt.close()

if ana == True:
    pers = []
    amps = []
    cnt = 0
    for i in range(len(solutions)):
        pers.append([])
        amps.append([])
        for j in range(len(solutions[i])):
            amp,per = uf.amp_freq(solutions[i][j],times)
            amps[i].append(amp)
            pers[i].append(per)
            cnt = cnt+1
            print('done: {} out of {}'.format(cnt,len(evo)*len(exp_sig)))

if img_freq == True:
    fig, axs = plt.subplots(3, 9)
    fig_ = plt.gcf()
    fig_.set_size_inches(30, 17.5)
    for i in range(len(solutions)):
        axs[int(i/9), i%9].plot(exp_sig, pers[i])
        axs[int(i/9), i%9].set_title(i)
    fig.savefig('Output/{}/periodicity'.format(in_file))
    plt.show()
        
    fig, axs = plt.subplots(3, 9)
    fig_ = plt.gcf()
    fig_.set_size_inches(30, 17.5)
    for i in range(len(solutions)):
        axs[int(i/9), i%9].plot(exp_sig, amps[i])
        axs[int(i/9), i%9].set_title(i)
    fig.savefig('Output/{}/amplitude'.format(in_file))
    plt.show()
        