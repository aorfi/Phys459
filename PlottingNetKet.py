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
# Organized in order N2M1,N2M2,N2M3.....,N2M10,N3M1,.....,N5M10
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

# Import Data
dataLocation = 'Data/06-23-20/CentralSpinN3M1.json'
saved = []
with open(dataLocation) as file:
    for line in file:
        saved.append(json.loads(line))
nkEngErr, nkRunTime = saved

# Import Comparison Data
dataLocation = 'Data/06-18-20/CSComparisonN2B1.json'
saved = []
with open(dataLocation) as file:
    for line in file:
        saved.append(json.loads(line))
engErrCS, engErr, runTimeCS, runTime = saved

#
# # Plotting Runs
# allEngErr = [engErrCS,engErr]
# allRunTime = [ runTimeCS,runTime]
# labels = ['NetKet Central Spin','Non-NetKet RBM']
# colors = ['blue', 'green']
#
# hisIt= np.arange(len(engErr))
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# ttl = plt.suptitle("Comparision N = 2, B = 1, M = 2 ",size =20)
# gs = gridspec.GridSpec(ncols=2, nrows=3, hspace = 0.4)
# ttl.set_position([.5, 0.92])
#
# ax2 = plt.subplot(gs[0, 0])
# ax2.hist(allEngErr, bins=10, color = colors, label=labels)
# ax2.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
#
# ax3 = plt.subplot(gs[0, 1])
# ax3.hist(allRunTime, bins=10, color = colors)
# ax3.set_xlabel("Runtime (s)",size = 15)
#
# ax4 = plt.subplot(gs[1, :])
# #ax4.scatter(hisIt,engErrHei, color = 'red', marker = '^', s=60)
# ax4.scatter(hisIt,engErrCS, color = 'blue', marker = '*', s=60)
# ax4.scatter(hisIt,engErr, color = 'green', marker = '>', s=60)
# #ax4.set_ylim([-0.000005,0.000005])
# ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)
#
# ax2.legend(labels, loc = (0.4, -3.3),fontsize = 12,ncol=3)
#
# ax5 = plt.subplot(gs[2, :])
# #ax5.scatter(hisIt,runTimeHei, color = 'red', marker = '^', s=60)
# ax5.scatter(hisIt,runTimeCS, color = 'blue', marker = '*', s=60)
# ax5.scatter(hisIt,runTime, color = 'green', marker = '>', s=60)
# ax5.set_xlabel("Run Number",size = 15)
# ax5 .set_ylabel("Runtime (s)", size = 15)
# plt.show()


index = 11
print(len(engErr))
allEngErr = [engErr[index],nkEngErr]
allRunTime = [time[index],nkRunTime]
labels = ['Non-NetKet RBM', 'NetKet RBM']
colors = ['blue', 'green']

hisIt= np.arange(len(engErr))
plt.figure(constrained_layout=True)
plt.figure(figsize=(8,8))
ttl = plt.suptitle("Comparision N = 2, B = 1, M = 2 ",size =20)
gs = gridspec.GridSpec(ncols=2, nrows=3, hspace = 0.4)
ttl.set_position([.5, 0.92])

ax2 = plt.subplot(gs[0, 0])
ax2.hist(allEngErr, bins=10, color = colors, label=labels)
ax2.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)

ax3 = plt.subplot(gs[0, 1])
ax3.hist(allRunTime, bins=10, color = colors)
ax3.set_xlabel("Runtime (s)",size = 15)