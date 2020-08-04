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
for i in range(4):
    for j in range(10):
        N = i + 2
        M = j + 1
        print('N,M', N,M)
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

# Plotting one
# index = 1
# hisIt= np.arange(lengths[index])
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(9,9))
# ttl = plt.suptitle("N = 2, M = 2, B = 1, A = 1 ",size =20)
# gs = gridspec.GridSpec(ncols=3, nrows=2, hspace = 0.3)
# ttl.set_position([.5, 0.93])
# ax1 = plt.subplot(gs[0, 0])
# ax1 .hist(stateErr[index], bins=10)
# ax1 .set_xlabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$",size = 15)
# ax1.set_ylabel("Frequency",size = 15)
# ax2 = plt.subplot(gs[0, 1])
# ax2.hist(engErr[index], bins=10)
# ax2.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
# ax3 = plt.subplot(gs[0, 2])
# ax3.hist(time[index], bins=10)
# ax3.set_xlabel("Runtime",size = 15)
# ax4 = plt.subplot(gs[1, :])
# ax4.scatter(hisIt,engErr[index])
# #ax4.set_ylim([-0.000005,0.000005])
# ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)
# ax5 = plt.subplot(gs[2, :])
# ax5.scatter(hisIt,time[index])
# ax5.set_xlabel("Run Number",size = 15)
# ax5 .set_ylabel("Runtime", size = 15)
# plt.show()

avEngErr = []
avStateErr = []
avRunTime = []
numFailure = []
numFailureErr = []
avEngErrF = []
avRunTimeF = []
numFailureF = []
numFailureErrF = []
cutOff = 0.1

for i in range(len(engErr)):
    avEngErrTemp = np.sum(engErr[i]) / (lengths[i])
    avEngErr.append(avEngErrTemp)
    avStateErrTemp = np.sum(stateErr[i]) / (lengths[i])
    avStateErr.append(avStateErrTemp)
    avRunTimeTemp = np.sum(time[i]) / (lengths[i])
    avRunTime.append(avRunTimeTemp)

    runsCutOff = sum(i > cutOff for i in engErr[i])
    numFailure.append(runsCutOff)
    numFailureErr.append(runsCutOff ** (0.5))

#
# listRunTime = [avRunTime[1],avRunTime[12],avRunTime[23],avRunTime[34]]
# xN = np.arange(2,6)
# xShort = np.arange(1,6)
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# ttl = plt.suptitle("Average Run Time with Increasing N \n" +r"$\alpha=1, B=1 , A=1$",size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.01)
# ttl.set_position([.5, 0.96])
#
# ax2 = plt.subplot(gs[0, 0])
# ax2.scatter(xN,listRunTime, color='r', s=50)
# ax2 .set_ylabel("Average RunTime", size = 15)
# ax2.set_xlabel("N",size = 15)
#
# plt.show()


#
# Import NetKet Data
dataLocation = "Data/08-04-20/NetKetN10M10B1.json"
saved = []
with open(dataLocation) as file:
    for line in file:
        saved.append(json.loads(line))
#engErrNK,engErrSR, engErr, stateErrNK, stateErrSR, stateErr, runTimeNK,runTimeSR, runTime= saved
engErrNK,engErrSR, stateErrNK, stateErrSR, runTimeNK,runTimeSR= saved

N=10
M=N
B=1
# Plotting
# allEngErr = [engErrNK,engErrSR, engErr]
# allStateErr = [stateErrNK,stateErrSR, stateErr]
# allRunTime = [ runTimeNK, runTimeSR, runTime]
# labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration', 'Non-NetKet RBM']
# colors = ['blue', 'green', 'red']
allEngErr = [engErrNK,engErrSR]
allStateErr = [stateErrNK,stateErrSR]
allRunTime = [ runTimeNK, runTimeSR]
labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration']
colors = ['blue', 'green']


hisIt= np.arange(len(engErrNK))
#plt.figure(constrained_layout=True)
plt.figure(figsize=(10,10))
ttl = plt.suptitle("Comparison of NetKet and Non-NetKet RBM \n N = " + str(N)+", B = "+str(B)+", M = " + str(M),size =20)
gs = gridspec.GridSpec(ncols=3, nrows=3, hspace = 0.4)
ttl.set_position([.5, 0.94])

ax1 = plt.subplot(gs[0, 0])
ax1.hist(allEngErr, bins=10, color = colors, label=labels)
ax1.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)

ax2 = plt.subplot(gs[0, 1])
ax2.hist(allStateErr, bins=10, color = colors, label=labels)
ax2.set_xlabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$",size = 15)

ax3 = plt.subplot(gs[0, 2])
ax3.hist(allRunTime, bins=10, color = colors)
ax3.set_xlabel("Runtime (s)",size = 15)

ax4 = plt.subplot(gs[1, :])
ax4.scatter(hisIt,engErrNK, color = 'blue')
ax4.scatter(hisIt,engErrSR, color = 'green',marker = '>')
#ax4.scatter(hisIt,engErr, color = 'red', marker = '^')
ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)

ax1.legend(labels, loc = (0, -3.3),fontsize = 12,ncol=3)

ax5 = plt.subplot(gs[2, :])
ax5.scatter(hisIt,runTimeNK, color = 'blue')
ax5.scatter(hisIt,runTimeSR, color = 'green',marker = '>')
#ax5.scatter(hisIt,runTime, color = 'red', marker = '^')
ax5.set_xlabel("Run Number",size = 15)
ax5 .set_ylabel("Runtime (s)", size = 15)
plt.show()




#
# labels = ['Gradient Descent','Stochastic Reconfiguration']
# NList= np.arange(len(engErrNK))
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# ttl = plt.suptitle("NetKet RBM \n N = 2-10, B = 1, " +r"$\alpha$ = 1",size =20)
# gs = gridspec.GridSpec(ncols=2, nrows=2, hspace = 0.2)
# ttl.set_position([.5, 0.94])
#
# ax1 = plt.subplot(gs[0, :])
# ax1.scatter(NList,engErrNK, color = 'blue', label = labels[0])
# #ax1.scatter(NList,engErrSR, color = 'green',marker = '>', label = labels[1])
# ax1.set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
#
# # ax4 = plt.subplot(gs[1, :])
# # ax4.scatter(NList,stateErrNK, color = 'blue')
# # ax4.scatter(NList,stateErrSR, color = 'green',marker = '>')
# # ax4 .set_ylabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$", size = 15)
#
# ax1.legend(labels, loc = (0, -3.3),fontsize = 12,ncol=3)
#
# ax5 = plt.subplot(gs[1, :])
# ax5.scatter(NList,runTimeNK, color = 'blue')
# #ax5.scatter(NList,runTimeSR, color = 'green',marker = '>')
# ax5.set_xlabel("Run Number",size = 15)
# ax5 .set_ylabel("Runtime (s)", size = 15)
# plt.show()
