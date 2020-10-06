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

# ****** Import Heisenberg Data ******
engErrHeiAll = []
stateErrHeiAll = []
runTimeHeiAll = []

NListHei = np.arange(2,10)
for i in range(len(NListHei)):
    N = NListHei[i]
    M=N
    dataLocation = "Data/09-15-20/HeiN"+str(N)+"M" + str(M)+".json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErrHei, stateErrHei, runTimeHei= saved
    engErrHeiAll.append(engErrHei)
    stateErrHeiAll.append(stateErrHei)
    runTimeHeiAll.append(runTimeHei)

# ****** Import Variable A Data ******
engErrVarAll = []
stateErrVarAll = []
runTimeVarAll = []

NListVar = np.arange(2,8)
for i in range(len(NListVar)):
    N = NListVar[i]
    M=N
    dataLocation = "Data/09-22-20/VarAN"+str(N)+"M" + str(M)+".json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErrSR, stateErrSR, runTimeSR= saved
    engErrVarAll.append(engErrSR)
    stateErrVarAll.append(stateErrSR)
    runTimeVarAll.append(runTimeSR)

# ****** Import CS Data ******
engErrSRAll = []
stateErrSRAll = []
runTimeSRAll = []

NListSR = np.arange(2, 12)
for i in range(len(NListSR)):
    N = NListSR[i]
    M = N
    dataLocation = "Data/08-14-20/nkN" + str(N) + "M" + str(M) + "B1.json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErrNK, engErrSR, stateErrNK, stateErrSR,runTimeNK, runTimeSR = saved
    engErrSRAll.append(engErrSR)
    stateErrSRAll.append(stateErrSR)
    runTimeSRAll.append(runTimeSR)


# **** Get averages *****
avEngErrHei = []
avStateErrHei = []
avRunTimeHei = []
cutOff = 0.01 #### *******
runsOverHei = []

avEngErrVar = []
avStateErrVar = []
avRunTimeVar = []
runsOverVar = []

avEngErrSR = []
avStateErrSR = []
avRunTimeSR = []
runsOverSR = []

for i in range(len(engErrHeiAll)):
    avEngErrTemp = np.sum(engErrHeiAll[i]) / (50)
    avEngErrHei.append(avEngErrTemp)
    avStateErrTemp = np.sum(stateErrHeiAll[i]) / (50)
    avStateErrHei.append(avStateErrTemp)
    avRunTimeTemp = np.sum(runTimeHeiAll[i]) / (50)
    avRunTimeHei.append(avRunTimeTemp)
    runsOverTemp = sum(j > cutOff for j in engErrHeiAll[i])
    runsOverHei.append(runsOverTemp)

for i in range(len(engErrVarAll)):
    avEngErrTemp = np.sum(engErrVarAll[i]) / (50)
    avEngErrVar.append(avEngErrTemp)
    avStateErrTemp = np.sum(stateErrVarAll[i]) / (50)
    avStateErrVar.append(avStateErrTemp)
    avRunTimeTemp = np.sum(runTimeVarAll[i]) / (50)
    avRunTimeVar.append(avRunTimeTemp)
    runsOverTemp = sum(j > cutOff for j in engErrVarAll[i])
    runsOverVar.append(runsOverTemp)

for i in range(len(engErrSRAll)):
    avEngErrSRTemp = np.sum(engErrSRAll[i]) / (50)
    avEngErrSR.append(avEngErrSRTemp)
    avStateErrSRTemp = np.sum(stateErrSRAll[i]) / (50)
    avStateErrSR.append(avStateErrSRTemp)
    avRunTimeSRTemp = np.sum(runTimeSRAll[i]) / (50)
    avRunTimeSR.append(avRunTimeSRTemp)
    runsOverSRTemp = sum(j > cutOff for j in engErrSRAll[i])
    runsOverSR.append(runsOverSRTemp)
#
# # ***** Histogram ****
# index = 5
# N = index+2
# M = N
# hisIt= np.arange(50)
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# ttl = plt.suptitle("Heisenberg Stochastic Reconfiguration \n N = " + str(N)+", M = " + str(M),size =20)
# gs = gridspec.GridSpec(ncols=3, nrows=3, hspace = 0.4)
# ttl.set_position([.5, 0.94])
# ax1 = plt.subplot(gs[0, 0])
# ax1.hist(engErrSRAll[index], bins=10)
# ax1.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
# ax2 = plt.subplot(gs[0, 1])
# ax2.hist(stateErrSRAll[index], bins=10)
# ax2.set_xlabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$",size = 15)
#
# ax3 = plt.subplot(gs[0, 2])
# ax3.hist(runTimeSRAll[index], bins=10)
# ax3.set_xlabel("Runtime (s)",size = 15)
#
# ax4 = plt.subplot(gs[1, :])
# ax4.scatter(hisIt,engErrSRAll[index])
# ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)
#
# ax5 = plt.subplot(gs[2, :])
# ax5.scatter(hisIt,runTimeSRAll[index])
# ax4.set_xlabel("Run Number",size = 15)
# ax5 .set_ylabel("Runtime (s)", size = 15)
# plt.show()
#

# #***** Run Time Scaling ******
#
#NRange= np.arange(2,len(avRunTimeSR)+2)
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
# labels = ['Central Spin with Constant A','Heisenberg', 'Central Spin with Variable A']
# colors = ['blue', 'green', 'red']
# ttl = plt.suptitle("Energy Error Scaling \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# #ax1.set_ylim(-0.2,1)
# ax1.scatter(NListSR, avEngErrSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(NListHei, avEngErrHei, color = colors[1], label=labels[1])
# ax1.scatter(NListVar, avEngErrVar, color = colors[2], label=labels[2], marker = '>')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Energy Error",size = 15)
# ax1.legend(labels, loc = (-0.1, -0.13),fontsize = 12,ncol=3)
# #ax1.set_yscale('log')
# plt.show()

# # # ***** State Error Scaling ******
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# labels = ['Central Spin with Constant A','Heisenberg', 'Central Spin with Variable A']
# colors = ['blue', 'green', 'red']
# ttl = plt.suptitle("State Error Scaling \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# #ax1.set_ylim(-0.2,1)
# ax1.scatter(NListSR, avStateErrSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(NListHei, avStateErrHei, color = colors[1], label=labels[1])
# ax1.scatter(NListVar, avStateErrVar, color = colors[2], label=labels[2], marker = '>')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average State Error",size = 15)
# ax1.legend(labels, loc = (-0.1, -0.13),fontsize = 12,ncol=3)
# #ax1.set_yscale('log')
# plt.show()

# # ***** Number of Runs******
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# labels = ['Central Spin with Constant A','Heisenberg', 'Central Spin with Variable A']
# colors = ['blue', 'green', 'red']
# ttl = plt.suptitle("Number of Runs with Energy Error above "+str(cutOff)+"\n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# #ax1.set_ylim(-0.2,1)
# ax1.scatter(NListSR, runsOverSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(NListHei, runsOverHei, color = colors[1], label=labels[1])
# ax1.scatter(NListVar, runsOverVar, color = colors[2], label=labels[2], marker = '>')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Number of Runs",size = 15)
# ax1.legend(labels, loc = (-0.1, -0.13),fontsize = 12,ncol=3)
# #ax1.set_yscale('log')
# plt.show()

# # ***** Average Run Time******
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# labels = ['Central Spin with Constant A','Heisenberg', 'Central Spin with Variable A']
# colors = ['blue', 'green', 'red']
# ttl = plt.suptitle("Average Run Time Scaling \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# #ax1.set_ylim(-0.2,1)
# ax1.scatter(NListSR, avRunTimeSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(NListHei,avRunTimeHei, color = colors[1], label=labels[1])
# ax1.scatter(NListVar, avRunTimeVar, color = colors[2], label=labels[2], marker = '>')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Run Time (s)",size = 15)
# ax1.legend(labels, loc = (-0.1, -0.13),fontsize = 12,ncol=3)
# ax1.set_yscale('log')
# plt.show()




fit = np.polyfit(np.log(NListSR), np.log(avRunTimeSR), 1, full=True)
fitPoints = (np.e**fit[0][1])*NListSR**fit[0][0]
print('fit: ', fit)
plt.figure(figsize=(8,8))
labels = ['Central Spin with Constant A','Heisenberg', 'Central Spin with Variable A']
colors = ['blue', 'green', 'red']
ttl = plt.suptitle("Average Run Time Scaling \n"+r"Central Spin with Constant A" ,size =20)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.97])
ax1 = plt.subplot(gs[0, 0])
#ax1.set_ylim(-0.2,1)
ax1.scatter(NListSR, avRunTimeSR, color = colors[0], label=labels[0], marker = '^')
ax1.plot(NListSR, fitPoints, color = colors[0], linestyle = '--')
#ax1.scatter(NListHei,avRunTimeHei, color = colors[1], label=labels[1])
#ax1.scatter(NListVar, avRunTimeVar, color = colors[2], label=labels[2], marker = '>')
ax1.set_xlabel("N",size = 15)
ax1.set_ylabel("Average Run Time (s)",size = 15)
#ax1.legend(labels, loc = (-0.1, -0.13),fontsize = 12,ncol=3)
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.show()

print('c= ', np.e**fit[0][1])
print('m= ', fit[0][0])