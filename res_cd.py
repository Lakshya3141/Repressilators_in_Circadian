"""
Code to obtain results section C and D
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
global hill
hill = 2

in_file = 'repC'
nodes,intermat,alpha,beta,basal = uf.adjacency(in_file)
init = np.random.rand(len(nodes)*2)*10
#init = np.array([0,0,900,900])

obs_node = 'C'
ind = np.where(nodes == obs_node)

t_fin = 100
dt_max = 0.01
steps = int(t_fin/dt_max)
times = np.linspace(0,t_fin,steps)
evo,source = uf.evol_reader(np.array(intermat),nodes,'sig')

exp_sig = np.linspace(10,10000,100)
exp_sig = [10,100,1000]
solutions = []
cnt = 0

for i,n in enumerate(evo[0:2]):
    solutions.append([])
    for e in exp_sig:
        mat = pd.DataFrame(evo[i],columns=nodes,index=source)
        sol = scip.solve_ivp(uf.diff_eq, (0,t_fin), y0 = init, t_eval=times,
                       args=(mat,alpha,beta,basal,hill,True,e))
        solutions[i].append(sol)
        cnt = cnt+1
        print("Done: ",cnt," out of ",len(evo)*len(exp_sig))









































'''
for n,nod in enumerate(nodes):
    plt.plot(sol.t,sol.y[2*n],label='p'+nod)
    #plt.plot(sol.t,sol.y[2*n+1],label='m'+nod)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Expression')
plt.show()

amp,per = uf.amp_freq(sol.y[0],sol.t)
for i in sol.y:
    print(uf.amp_freq(i,sol.t))

### FFT of protein A expression ###
sig_noise_fft = fft(sol.y[0])
sig_noise_amp = 2 / sol.t.size * np.abs(sig_noise_fft)
sig_noise_freq = np.abs(fftfreq(sol.t.size, dt_max))
sig_noise_amp[0] = 0

plt.plot(sig_noise_freq,sig_noise_amp)
plt.ylabel('Amplitude')
plt.xlabel('Frequency')


periodicity = 1/sig_noise_freq[np.argmax(sig_noise_amp)]
print(periodicity)
'''
