"""
Central Connect of all snippets
@author: Lakshya Chauhan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time,random
import util_funcs as uf
global nodes, intermat
import scipy.integrate as scip
global hill
hill = 2
"""
if sys.argv[1]:
    in_file = sys.argv[1] #Name of input
else:
    in_file = 'rep' #Or it takes in the default file 'rep'
"""  
in_file = 'rep'
nodes,intermat,alpha,beta,basal = uf.adjacency(in_file)
init = np.random.rand(len(nodes)*2)*100
#init = np.array([0,0,900,900])

t_fin = 50
sol = scip.solve_ivp(uf.diff_eq,(0,t_fin),y0 = init,
                       args=(intermat,alpha,beta,basal,hill))

for n,nod in enumerate(nodes):
    plt.plot(sol.t,sol.y[2*n],label='p'+nod)
    #plt.plot(sol.t,sol.y[2*n+1],label='m'+nod)
plt.legend()
print(intermat)
plt.show()

