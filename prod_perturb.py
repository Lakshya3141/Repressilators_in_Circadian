"""
trial code for obtaining various paramaters
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

in_file = 'quadRep'
nodes,intermat,alpha,beta,basal = uf.adjacency(in_file)
init = np.random.rand(len(nodes)*2)*10
#init = np.array([0,0,900,900])

t_fin = 100
dt_max = 0.01
steps = int(t_fin/dt_max)
times = np.linspace(0,t_fin,steps)
solutions = []
amp = []
per = []
alp = np.linspace(1,20000,50)
for i in range(50):
    alpha[0] = alp[i]
    sol = scip.solve_ivp(uf.diff_eq, (0,t_fin), y0 = init, t_eval=times,
                       args=(intermat,alpha,beta,basal,hill))
    a = []
    p = []
    for j in range(len(nodes)):
        amps,pers = uf.amp_freq(sol.y[2*j+1],sol.t)
        a.append(amps)
        p.append(pers)
    amp.append(a)
    per.append(p)
    solutions.append(sol.y)
    print('done:',i)


for i in range(3):
    plt.plot(alp,a[i],label='p'+nodes[i])
plt.title('amplitude vs production of A')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('amplitudes')

# amp,per = uf.amp_freq(sol.y[0],sol.t)
# for i in sol.y:
#     print(uf.amp_freq(i,sol.t))

