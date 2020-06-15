import json
from qutip import *
import numpy as np
import scipy
import cmath
import matplotlib.pyplot as plt
import itertools
from numpy.random import seed
from numpy.random import rand
from itertools import product
from scipy.misc import derivative
from scipy import optimize
from functools import wraps
from time import time
from scipy.optimize import minimize
from matplotlib import gridspec
plt.style.use('seaborn')

# Import previous RBM Data
engErr = []
stateErr = []
time = []
lengths = []
for i in range(3):
    for j in range(10):
        N = i + 2
        M = j + 1
        dataLocation = 'Data/06-08-20/OneRunN' + str(N) + 'M' + str(M) + '.json'
        saved = []
        with open(dataLocation) as file:
            for line in file:
                saved.append(json.loads(line))

        cgdTime, cgdEngErr, cgdStateErr, edTime, length = saved
        engErr.append(cgdEngErr)
        stateErr.append(cgdStateErr)
        time.append(cgdTime)
        lengths.append(length)

# Import NetKet Central Spin Data
dataLocation = 'Data/06-15-20/CentralSpinN3M3.json'
saved = []
with open(dataLocation) as file:
    for line in file:
        saved.append(json.loads(line))
nkCentralEngErr, nkCentralRunTime = saved

# Import NetKet Heisenberg Data
dataLocation = 'Data/06-15-20/HeiN2.json'
saved = []
with open(dataLocation) as file:
    for line in file:
        saved.append(json.loads(line))
nkHeisenEngErr, nkHeisenRunTime = saved
print(len(nkHeisenEngErr))




# Plotting
index = 1
#allEngErr = [nkCentralEngErr,engErr[index],nkHeisenEngErr]
#allRunTime = [nkCentralRunTime,time[index],nkHeisenRunTime]

#labels = ['NetKet Central Spin', 'RBM', 'NetKet Heisenberg']
#colors = ['red','blue', 'green']

hisIt= np.arange(len(nkCentralEngErr))
plt.figure(constrained_layout=True)
plt.figure(figsize=(10,10))
ttl = plt.suptitle("Heisenberg Spin Chain N = 2",size =20)
gs = gridspec.GridSpec(ncols=2, nrows=3, hspace = 0.4)
ttl.set_position([.5, 0.92])

ax2 = plt.subplot(gs[0, 0])
#ax2.hist(allEngErr, bins=10, color = colors, label=labels)
ax2.hist(nkHeisenEngErr, bins=10, color = 'red')
ax2.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)

ax3 = plt.subplot(gs[0, 1])
#ax3.hist(allRunTime, bins=10, color = colors)
ax3.hist(nkHeisenRunTime, bins=10, color = 'red')
ax3.set_xlabel("Runtime (s)",size = 15)

ax4 = plt.subplot(gs[1, :])
ax4.scatter(hisIt,nkHeisenEngErr, color = 'red')
#ax4.scatter(hisIt,engErr[index], color = 'blue')
#ax4.scatter(hisIt,nkHyEngErr, color = 'green')
#ax4.set_ylim([-0.000005,0.000005])
ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)

#ax2.legend(labels, loc = (0.4, -3.3),fontsize = 12,ncol=3)

ax5 = plt.subplot(gs[2, :])
ax5.scatter(hisIt,nkHeisenRunTime, color = 'red')
#ax5.scatter(hisIt,time[index], color = 'blue')
#ax5.scatter(hisIt,nkHyRunTime, color = 'green')
ax5.set_xlabel("Run Number",size = 15)
ax5 .set_ylabel("Runtime (s)", size = 15)
plt.show()










