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


# ********* This document isn't organized as I use it to store the plotting templates


# ****** Import Exact Data ******
RunTimeExact = []
NListExact = np.arange(2,14)
for i in range(len(NListExact)):
    N = NListExact[i]
    M=N
    dataLocation = "Data/21-02-16/exactRunTimeN"+str(N)+"M" + str(M)+".json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    runTime= saved
    RunTimeExact.append(runTime)
# ****** Import Exact Data ******
RunTimeExactVar = []
NListExactVar = np.arange(2,14)
for i in range(len(NListExactVar)):
    N = NListExactVar[i]
    M=N
    dataLocation = "Data/21-02-16/varexactRunTimeN"+str(N)+"M" + str(M)+".json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    runTime= saved
    RunTimeExactVar.append(runTime)


##****** Import Variable A Data ******
engErrVarAll = []
stateErrVarAll = []
runTimeVarAll = []
#
NListVar = np.arange(2,14)
for i in range(len(NListVar)):
    N = NListVar[i]
    M=N
    dataLocation = "Data/21-02-16/varN"+str(N)+"M" + str(M)+".json"
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

NListSR = np.arange(2, 14)
for i in range(len(NListSR)):
    N = NListSR[i]
    M = N
    dataLocation = "Data/21-02-16/N" + str(N) + "M" + str(M) + ".json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErrSR, stateErrSR, runTimeSR = saved
    engErrSRAll.append(engErrSR)
    stateErrSRAll.append(stateErrSR)
    runTimeSRAll.append(runTimeSR)


# **** Get averages *****
cutOff = 0.01 #### *******

avEngErrHeiF = []
avStateErrHeiF = []
avRunTimeHeiF = []
runsOverHeiF = []

avEngErrVar = []
avStateErrVar = []
avRunTimeVar = []
runsOverVar = []

avEngErrSR = []
avStateErrSR = []
avRunTimeSR = []
runsOverSR = []

# heiRem = 0
# for i in range(len(engErrHeiFAll)):
#     #remove nan from averages
#     engErrHeiFAllClean = [x for x in engErrHeiFAll[i] if str(x) != 'nan']
#     heiRem += len(engErrHeiFAll[i])- len(engErrHeiFAllClean)
#     avEngErrTemp = np.sum(engErrHeiFAllClean) / (50)
#     avEngErrHeiF.append(avEngErrTemp)
#     stateErrHeiFAllClean = [x for x in stateErrHeiFAll[i] if str(x) != 'nan']
#     avStateErrTemp = np.sum(stateErrHeiFAllClean) / (50)
#     avStateErrHeiF.append(avStateErrTemp)
#     avRunTimeTemp = np.sum(runTimeHeiFAll[i]) / (50)
#     avRunTimeHeiF.append(avRunTimeTemp)
#     runsOverTemp = sum(j > cutOff for j in engErrHeiFAll[i])
#     runsOverHeiF.append(runsOverTemp)
# print('Number Removed Hei', heiRem)
#
varRem = 0
for i in range(len(engErrVarAll)):
    # remove nan from averages
    engErrVarAllClean = [x for x in engErrVarAll[i] if str(x) != 'nan']
    varRem += len(engErrVarAll[i]) - len(engErrVarAllClean)
    avEngErrTemp = np.sum(engErrVarAllClean) / len(NListVar)
    avEngErrVar.append(avEngErrTemp)
    stateErrVarAllClean = [x for x in stateErrVarAll[i] if str(x) != 'nan']
    avStateErrTemp = np.sum(stateErrVarAllClean) / len(NListVar)
    avStateErrVar.append(avStateErrTemp)
    avRunTimeTemp = np.sum(runTimeVarAll[i]) / len(NListVar)
    avRunTimeVar.append(avRunTimeTemp)
    runsOverTemp = sum(j > cutOff for j in engErrVarAll[i])
    runsOverVar.append(runsOverTemp)
print('Number Removed var', varRem)

csRem = 0
for i in range(len(engErrSRAll)):
    #remove nan from averages
    engErrSRAllClean = [x for x in engErrSRAll[i] if str(x) != 'nan']
    csRem += len(engErrSRAll[i]) - len(engErrSRAllClean)
    avEngErrSRTemp = np.sum(engErrSRAllClean) / len(NListSR)
    avEngErrSR.append(avEngErrSRTemp)
    stateErrSRAllClean = [x for x in stateErrSRAll[i] if str(x) != 'nan']
    avStateErrSRTemp = np.sum(stateErrSRAllClean) / len(NListSR)
    avStateErrSR.append(avStateErrSRTemp)
    avRunTimeSRTemp = np.sum(runTimeSRAll[i]) / len(NListSR)
    avRunTimeSR.append(avRunTimeSRTemp)
    runsOverSRTemp = sum(j > cutOff for j in engErrSRAll[i])
    runsOverSR.append(runsOverSRTemp)
print('Number Removed cs', csRem)
#
##***** Histogram ****
# index = 6
# N = index+2
# M = N
# hisIt= np.arange(50)
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# engErr = [engErrSRAll[index],engErrVarAll[index]]#,engErrHeiFAll[index]]
# stateErr= [stateErrSRAll[index],stateErrVarAll[index]]#,stateErrHeiFAll[index]]
# runTime= [runTimeSRAll[index],runTimeVarAll[index]]#,runTimeHeiFAll[index]]
#
# labels = ['Central Spin with Constant A','Central Spin with Varying A']#,'Heisenberg with Field']
# colors = ['blue', 'green']#, 'red']
# ttl = plt.suptitle("Stochastic Reconfiguration with"+ r" $\alpha = 1$"+" \n N = " + str(N)+", M = " + str(M),size =20)
# gs = gridspec.GridSpec(ncols=3, nrows=4, hspace = 0.4)
# ttl.set_position([.5, 0.94])
# ax1 = plt.subplot(gs[0, 0])
# ax1.hist(engErr, bins=10, color = colors, label=labels)
# ax1.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
# ax2 = plt.subplot(gs[0, 1])
# ax2.hist(stateErr, bins=10, color = colors, label=labels)
# ax2.set_xlabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$",size = 15)
#
# ax3 = plt.subplot(gs[0, 2])
# ax3.hist(runTime, bins=10, color = colors, label=labels)
# ax3.set_xlabel("Runtime (s)",size = 15)
#
# ax4 = plt.subplot(gs[1, :])
# ax4.scatter(hisIt, engErrSRAll[index], color = colors[0], label=labels[0], marker = '^')
# ax4.scatter(hisIt, engErrVarAll[index], color = colors[1], label=labels[1], marker = '>')
# #ax4.scatter(hisIt, engErrHeiFAll[index], color = colors[2], label=labels[2], marker = '<')
# ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)
#
# ax4 = plt.subplot(gs[2, :])
# ax4.scatter(hisIt, stateErrSRAll[index], color = colors[0], label=labels[0], marker = '^')
# ax4.scatter(hisIt, stateErrVarAll[index], color = colors[1], label=labels[1], marker = '>')
# #ax4.scatter(hisIt, stateErrHeiFAll[index], color = colors[2], label=labels[2], marker = '<')
# ax4 .set_ylabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$", size = 15)
#
# ax5 = plt.subplot(gs[3, :])
# ax5.scatter(hisIt, runTimeSRAll[index], color = colors[0], label=labels[0], marker = '^')
# ax5.scatter(hisIt, runTimeVarAll[index], color = colors[1], label=labels[1], marker = '>')
# #ax5.scatter(hisIt, runTimeHeiFAll[index], color = colors[2], label=labels[2], marker = '<')
# ax5.set_xlabel("Run Number",size = 15)
# ax5 .set_ylabel("Runtime (s)", size = 15)
# ax5.legend(labels, loc = (0.1, -0.6),fontsize = 12,ncol=3)
# plt.show()
# # #
#
# # #***** Run Time Scaling ******
#
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# ttl = plt.suptitle("Runtime Scaling \n"+r" $\alpha = 1$" ,size =18)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.98])
# labels = ['RBM Constant A','RBM Varying A','ED Constant A','ED Varying A']
# colors = ['blue', 'green', 'red']
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(NListSR, avRunTimeSR, color = colors[0], label=labels[0], marker = '^')
# #ax1.scatter(NListVar, avRunTimeVar, color = colors[1], label=labels[1], marker = '>')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Runtime (s)",size = 15)
# #ax2 = ax1.twinx()
# #ax1.scatter(NListExact, RunTimeExact, color = colors[2], label=labels[2], marker = '>')
# #ax1.scatter(NListExactVar, RunTimeExactVar, color = 'black', label=labels[3], marker = '>')
# ax1.legend(loc = (-0.1, -0.13),fontsize = 12,ncol=4)
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# plt.show()

# ***** Energy Error Scaling ******
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# labels = ['RBM Constant A', 'RBM Varying A','Heisenberg with Field']
# colors = ['blue', 'green', 'red']
# ttl = plt.suptitle("Energy Error Scaling \n"+r"$\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# #ax1.set_ylim(-0.2,1)
# ax1.scatter(NListSR, avEngErrSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(NListVar, avEngErrVar, color = colors[1], label=labels[1], marker = '>')
# #ax1.scatter(NListHeiF, avEngErrHeiF, color = colors[2], label=labels[2], marker = '<')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Energy Error",size = 15)
# ax1.legend(labels, loc = (0, -0.13),fontsize = 12,ncol=3)
# ax1.set_yscale('log')
# plt.show()




# ***** State Error Scaling ******
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# labels = ['RBM Constant A', 'RBM Varying A','Heisenberg with Field']
# colors = ['blue', 'green', 'red']
# ttl = plt.suptitle("State Error Scaling \n"+r"$\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# #ax1.set_ylim(-0.2,1)
# ax1.scatter(NListSR, avStateErrSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(NListVar, avStateErrVar, color = colors[1], label=labels[1], marker = '>')
# #ax1.scatter(NListHeiF, avStateErrHeiF, color = colors[2], label=labels[2], marker = '<')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average State Error",size = 15)
# ax1.legend(labels, loc = (0, -0.13),fontsize = 12,ncol=3)
# #ax1.set_yscale('log')
# plt.show()

# # # ***** Number of Runs******
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# labels = ['Central Spin with Constant A', 'Central Spin with Varying A']#,'Heisenberg with Field']
# colors = ['blue', 'green']#, 'red']
# ttl = plt.suptitle("Number of Runs with Energy Error above "+str(cutOff)+"\n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# #ax1.set_ylim(-0.2,1)
# ax1.scatter(NListSR, runsOverSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(NListVar, runsOverVar, color = colors[1], label=labels[1], marker = '>')
# #ax1.scatter(NListHeiF, runsOverHeiF, color = colors[2], label=labels[2], marker = '<')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Number of Runs",size = 15)
# ax1.legend(labels, loc = (0, -0.13),fontsize = 12,ncol=3)
# #ax1.set_yscale('log')
# plt.show()

# # # ***** Average Run Time******
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# labels = ['Central Spin with Constant A', 'Central Spin with Varying A','Heisenberg with Field']
# colors = ['blue', 'green', 'red']
# ttl = plt.suptitle("Average Run Time \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# #ax1.set_ylim(-0.2,1)
# ax1.scatter(NListSR, avRunTimeSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(NListVar, avRunTimeVar, color = colors[1], label=labels[1], marker = '>')
# #ax1.scatter(NListHeiF, avRunTimeHeiF, color = colors[2], label=labels[2], marker = '<')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Run Time (s)",size = 15)
# ax1.legend(labels, loc = (0, -0.13),fontsize = 12,ncol=3)
# ax1.set_yscale('log')
# plt.show()




x = np.arange(2,13,0.01)
NListSRno2= np.delete(NListSR,0)
avRunTimeSRno2= np.delete(avRunTimeSR,0)
fit = np.polyfit(np.log10(NListSR), np.log10(avRunTimeSR), 1, full=True)
fitPoints = (10**fit[0][1])*x**fit[0][0]

NListExactCut = NListExact[5:]
RunTimeExactCut = RunTimeExact[5:]
fitExact = np.polyfit(NListExactCut, np.log10(RunTimeExactCut), 1, full=True)
fitPointsExact = (10**fitExact[0][1])*(10**fitExact[0][0])**x

NListVarno2= np.delete(NListVar,0)
avRunTimeVarno2= np.delete(avRunTimeVar,0)
fitVar = np.polyfit(np.log10(NListVarno2), np.log10(avRunTimeVarno2), 1, full=True)
fitPointsVar = (10**fitVar[0][1])*x**fitVar[0][0]

NListExactVarCut = NListExact[6:]
RunTimeExactVarCut = RunTimeExact[6:]
fitExactVar = np.polyfit(NListExactVarCut, np.log10(RunTimeExactVarCut), 1, full=True)
fitPointsExactVar = (10**fitExactVar[0][1])*(10**fitExactVar[0][0])**x


plt.figure(figsize=(8,8))
labels = ['RBM Constant A','RBM Varying A','ED Constant A','ED Varying A']
colors = ['blue', 'green', 'red','black']
ttl = plt.suptitle("Average Run Time Scaling " ,size =20)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.97])
ax1 = plt.subplot(gs[0, 0])
#ax1.set_ylim(-0.2,1)

ax1.scatter(NListSR, avRunTimeSR, color = colors[0], label=labels[0], marker = '^')
ax1.plot(x, fitPoints, color = colors[0], linestyle = '--')
# ax1.scatter(NListExact, RunTimeExact, color = colors[2], label=labels[2], marker = '>')
# ax1.plot(x, fitPointsExact, color = colors[2], linestyle = '--')
#
# ax1.scatter(NListVar, avRunTimeVar, color = colors[1], label=labels[1], marker = '>')
# ax1.plot(x, fitPointsVar, color = colors[1], linestyle = '--')
# ax1.scatter(NListExactVar, RunTimeExactVar, color = colors[3], label=labels[3], marker = '>')
# ax1.plot(x, fitPointsExactVar, color = colors[3], linestyle = '--')

ax1.set_xlabel("N",size = 15)
ax1.set_ylabel("Average Run Time (s)",size = 15)
#ax1.legend(labels, loc = (-0.1, -0.13),fontsize = 12,ncol=3)
#ax1.set_yscale('log')
#ax1.set_xscale('log')
ax1.legend(loc = (-0.1, -0.13),fontsize = 12,ncol=4)
plt.show()

print('a= ', 10**fit[0][1])
print('k= ', fit[0][0])

print('lamba= ', 10**fitExact[0][1])
print('b= ', 10**fitExact[0][0])

print('a= ', 10**fitVar[0][1])
print('k= ', fitVar[0][0])

print('lamba= ', 10**fitExactVar[0][1])
print('b= ', 10**fitExactVar[0][0])