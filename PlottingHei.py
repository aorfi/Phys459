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

# ****** Import NetKet Data ******
engErrSRAll = []
stateErrSRAll = []
runTimeSRAll = []

NList = np.arange(2,10)
for i in range(len(NList)):
    N = NList[i]
    M=N
    dataLocation = "Data/09-15-20/HeiN"+str(N)+"M" + str(M)+".json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErrSR, stateErrSR, runTimeSR= saved
    engErrSRAll.append(engErrSR)
    stateErrSRAll.append(stateErrSR)
    runTimeSRAll.append(runTimeSR)

# **** Get averages *****
avEngErrSR = []
avStateErrSR = []
avRunTimeSR = []
cutOff = 0.01 #### *******
runsOverSR = []

for i in range(len(engErrSRAll)):
    avEngErrTemp = np.sum(engErrSRAll[i]) / (50)
    avEngErrSR.append(avEngErrTemp)
    avStateErrTemp = np.sum(stateErrSRAll[i]) / (50)
    avStateErrSR.append(avStateErrTemp)
    avRunTimeTemp = np.sum(runTimeSRAll[i]) / (50)
    avRunTimeSR.append(avRunTimeTemp)
    runsOverTemp = sum(j > cutOff for j in engErrSRAll[i])
    runsOverSR.append(runsOverTemp)

print(avEngErrSR)

# ***** Histogram ****
index = 5
N = index+2
M = N
hisIt= np.arange(50)
#plt.figure(constrained_layout=True)
plt.figure(figsize=(10,10))
ttl = plt.suptitle("Heisenberg Stochastic Reconfiguration \n N = " + str(N)+", M = " + str(M),size =20)
gs = gridspec.GridSpec(ncols=3, nrows=3, hspace = 0.4)
ttl.set_position([.5, 0.94])
ax1 = plt.subplot(gs[0, 0])
ax1.hist(engErrSRAll[index], bins=10)
ax1.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
ax2 = plt.subplot(gs[0, 1])
ax2.hist(stateErrSRAll[index], bins=10)
ax2.set_xlabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$",size = 15)

ax3 = plt.subplot(gs[0, 2])
ax3.hist(runTimeSRAll[index], bins=10)
ax3.set_xlabel("Runtime (s)",size = 15)

ax4 = plt.subplot(gs[1, :])
ax4.scatter(hisIt,engErrSRAll[index])
ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)

ax5 = plt.subplot(gs[2, :])
ax5.scatter(hisIt,runTimeSRAll[index])
ax4.set_xlabel("Run Number",size = 15)
ax5 .set_ylabel("Runtime (s)", size = 15)
plt.show()
#
# #***** Run Time Scaling ******
#
NRange= np.arange(2,len(avRunTimeSR)+2)
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(6,6))
# ttl = plt.suptitle("Runtime Scaling Heisenberg \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =18)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.98])
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(NRange, avRunTimeSR)
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Runtime (s)",size = 15)
# #ax1.set_yscale('log')
# #ax1.set_xscale('log')
# plt.show()
#
# # # ***** Energy Error Scaling ******
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# ttl = plt.suptitle("Energy Error Scaling Heisenberg \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(NRange, avEngErrSR)
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Energy Error",size = 15)
# #ax1.set_yscale('log')
# plt.show()
#
# # # ***** State Error Scaling ******
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# ttl = plt.suptitle("State Error Scaling Heisenberg \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(NRange, avStateErrSR)
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average State Error",size = 15)
# #ax1.set_yscale('log')
# plt.show()
# #
# # # # ***** Number  of runs******
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# ttl = plt.suptitle("Number of Runs with Energy Error above "+str(cutOff) +r" $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(NRange, runsOverSR)
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Number of Runs",size = 15)
# plt.show()
