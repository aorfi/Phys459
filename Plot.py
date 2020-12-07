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


# ****** Import CS Data ******
engErrSRAll = []
stateErrSRAll = []
runTimeSRAll = []

NListSR = np.arange(2, 5)
for i in range(len(NListSR)):
    N = NListSR[i]
    M = N
    dataLocation = "Data/12-01-20/csN" + str(N) + "M" + str(M) + ".json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErrSR, stateErrSR, runTimeSR = saved
    engErrSRAll.append(engErrSR)
    stateErrSRAll.append(stateErrSR)
    runTimeSRAll.append(runTimeSR)


engErrAll = []
stateErrAll = []
runTimeAll = []
NListSR = np.arange(2, 5)
for i in range(len(NListSR)):
    N = NListSR[i]
    M = N
    dataLocation = "Data/12-01-20/varAN" + str(N) + "M" + str(M) + ".json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErr, stateErr, runTime = saved
    engErrAll.append(engErr)
    stateErrAll.append(stateErr)
    runTimeAll.append(runTime)

##***** Histogram ****
index = 2
N = index+2
M = N
hisIt= np.arange(len(engErrSRAll[index]))
#plt.figure(constrained_layout=True)
plt.figure(figsize=(10,10))
engErr = [engErrSRAll[index],engErrAll[index]]
stateErr= [stateErrSRAll[index],stateErrAll[index]]
runTime= [runTimeSRAll[index],runTimeAll[index]]

labels = ['Constant A','Varying A']
colors = ['blue', 'green']
ttl = plt.suptitle("Stochastic Reconfiguration \n N = " + str(N)+", M = " + str(M),size =20)
gs = gridspec.GridSpec(ncols=3, nrows=4, hspace = 0.4)
ttl.set_position([.5, 0.94])
ax1 = plt.subplot(gs[0, 0])
ax1.hist(engErr, bins=10, color = colors, label=labels)
ax1.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
ax2 = plt.subplot(gs[0, 1])
ax2.hist(stateErr, bins=10, color = colors, label=labels)
ax2.set_xlabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$",size = 15)

ax3 = plt.subplot(gs[0, 2])
ax3.hist(runTime, bins=10, color = colors, label=labels)
ax3.set_xlabel("Runtime (s)",size = 15)

ax4 = plt.subplot(gs[1, :])
ax4.scatter(hisIt, engErrSRAll[index], color = colors[0], label=labels[0], marker = '^')
ax4.scatter(hisIt, engErrAll[index], color = colors[1], label=labels[1], marker = '>')
ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)

ax4 = plt.subplot(gs[2, :])
ax4.scatter(hisIt, stateErrSRAll[index], color = colors[0], label=labels[0], marker = '^')
ax4.scatter(hisIt, stateErrAll[index], color = colors[1], label=labels[1], marker = '>')
ax4 .set_ylabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$", size = 15)

ax5 = plt.subplot(gs[3, :])
ax5.scatter(hisIt, runTimeSRAll[index], color = colors[0], label=labels[0], marker = '^')
ax5.scatter(hisIt, runTimeAll[index], color = colors[1], label=labels[1], marker = '>')
ax5.set_xlabel("Run Number",size = 15)
ax5 .set_ylabel("Runtime (s)", size = 15)
ax5.legend(labels, loc = (0, -0.5),fontsize = 12,ncol=3)
plt.show()

#
# params = []
# N=2
# fileName = "Data/12-01-20/ParametersN"+str(N)+"M" + str(N)+".json"
# saved = []
# with open(fileName) as file:
#     for line in file:
#         saved.append(json.loads(line))
# paramAll, paramAllNK, paramAllOutput = saved
# params.append(paramAllOutput)
# N=4
# fileName = "Data/12-01-20/ParametersN"+str(N)+"M" + str(N)+".json"
# saved = []
# with open(fileName) as file:
#     for line in file:
#         saved.append(json.loads(line))
# paramAll, paramAllNK, paramAllOutput = saved
# params.append(paramAllOutput)
# N=6
# fileName = "Data/12-01-20/ParametersN"+str(N)+"M" + str(N)+".json"
# saved = []
# with open(fileName) as file:
#     for line in file:
#         saved.append(json.loads(line))
# paramAll, paramAllNK, paramAllOutput = saved
# params.append(paramAllOutput)
# N=8
# fileName = "Data/12-01-20/ParametersN"+str(N)+"M" + str(N)+".json"
# saved = []
# with open(fileName) as file:
#     for line in file:
#         saved.append(json.loads(line))
# paramAll, paramAllNK, paramAllOutput = saved
# params.append(paramAllOutput)
#
# cutParams = []
# for i in range(4):
#     cut = params[i]
#     cut = cut[0:320]
#     cutParams.append(cut)
#
# num = len(paramAllNK)
# plt.figure(figsize=(10,10))
# labels = ['N=2','N=4','N=6','N=8']
# colors = ['blue', 'green', 'red', 'black']
# ttl = plt.suptitle("Ground State RBM Parameters " ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.94])
# ax1 = plt.subplot(gs[0, 0])
# ax1.hist(cutParams, bins=20, color = colors, label=labels)
# ax1.set_xlim(-2,2)
# ax1.legend(labels, loc = (0.2, -0.1),fontsize = 12,ncol=4)
# plt.show()