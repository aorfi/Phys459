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
# RunTimeExact = []
# NListExact = np.arange(2,14)
# for i in range(len(NListExact)):
#     N = NListExact[i]
#     M=N
#     dataLocation = "Data/21-02-09/exactRunTimeN"+str(N)+"M" + str(M)+".json"
#     saved = []
#     with open(dataLocation) as file:
#         for line in file:
#             saved.append(json.loads(line))
#     runTime= saved
#     RunTimeExact.append(runTime)
# # ****** Import Exact Data ******
# RunTimeExactVar = []
# NListExactVar = np.arange(2,14)
# for i in range(len(NListExactVar)):
#     N = NListExactVar[i]
#     M=N
#     dataLocation = "Data/21-02-09/varexactRunTimeN"+str(N)+"M" + str(M)+".json"
#     saved = []
#     with open(dataLocation) as file:
#         for line in file:
#             saved.append(json.loads(line))
#     runTime= saved
#     RunTimeExactVar.append(runTime)
#

##****** Import Variable A Data ******
engErrVarAll = []
stateErrVarAll = []
runTimeVarAll = []
#
aListVar = np.arange(1,5)
for i in range(len(aListVar)):
    a = aListVar[i]
    N=7
    M=a*N
    dataLocation = "Data/21-02-09/varalpha"+str(a)+"N"+str(N)+"M" + str(M)+".json"
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

aListSR = np.arange(1, 5)
for i in range(len(aListSR)):
    a = aListSR[i]
    N=7
    M = a*N
    dataLocation = "Data/21-02-09/alpha"+str(a)+"N" + str(N) + "M" + str(M) + ".json"
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
    avEngErrTemp = np.sum(engErrVarAllClean) / 50
    avEngErrVar.append(avEngErrTemp)
    stateErrVarAllClean = [x for x in stateErrVarAll[i] if str(x) != 'nan']
    avStateErrTemp = np.sum(stateErrVarAllClean) / 50
    avStateErrVar.append(avStateErrTemp)
    avRunTimeTemp = np.sum(runTimeVarAll[i]) / 50
    avRunTimeVar.append(avRunTimeTemp)
    runsOverTemp = sum(j > cutOff for j in engErrVarAll[i])
    runsOverVar.append(runsOverTemp)
print('Number Removed var', varRem)

csRem = 0
for i in range(len(engErrSRAll)):
    #remove nan from averages
    engErrSRAllClean = [x for x in engErrSRAll[i] if str(x) != 'nan']
    csRem += len(engErrSRAll[i]) - len(engErrSRAllClean)
    avEngErrSRTemp = np.sum(engErrSRAllClean) / 50
    avEngErrSR.append(avEngErrSRTemp)
    stateErrSRAllClean = [x for x in stateErrSRAll[i] if str(x) != 'nan']
    avStateErrSRTemp = np.sum(stateErrSRAllClean) / 50
    avStateErrSR.append(avStateErrSRTemp)
    avRunTimeSRTemp = np.sum(runTimeSRAll[i]) / 50
    avRunTimeSR.append(avRunTimeSRTemp)
    runsOverSRTemp = sum(j > cutOff for j in engErrSRAll[i])
    runsOverSR.append(runsOverSRTemp)
print('Number Removed cs', csRem)
#
##***** Histogram ****
# index = 3
# a = index+1
# N = 7
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
# ttl = plt.suptitle("Stochastic Reconfiguration with"+ r" $\alpha = $"+str(a)+" \n N = " + str(N)+", M = " + str(M),size =20)
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
plt.figure(constrained_layout=True)
plt.figure(figsize=(8,8))
ttl = plt.suptitle("Runtime Scaling" ,size =18)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.98])
labels = ['RBM Constant A','RBM Varying A','ED Constant A','ED Varying A']
colors = ['blue', 'green', 'red']
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(aListSR, avRunTimeSR, color = colors[0], label=labels[0], marker = '^')
ax1.scatter(aListVar, avRunTimeVar, color = colors[1], label=labels[1], marker = '>')
ax1.set_xlabel("Alpha",size = 15)
ax1.set_ylabel("Average Runtime (s)",size = 15)
#ax2 = ax1.twinx()
#ax1.scatter(NListExact, RunTimeExact, color = colors[2], label=labels[2], marker = '>')
#ax1.scatter(NListExactVar, RunTimeExactVar, color = 'black', label=labels[3], marker = '>')
ax1.legend(loc = (-0.1, -0.13),fontsize = 12,ncol=4)
# ax1.set_yscale('log')
# ax1.set_xscale('log')
plt.show()

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
# ax1.scatter(aListSR, avEngErrSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(aListVar, avEngErrVar, color = colors[1], label=labels[1], marker = '>')
# #ax1.scatter(NListHeiF, avEngErrHeiF, color = colors[2], label=labels[2], marker = '<')
# ax1.set_xlabel("Alpha",size = 15)
# ax1.set_ylabel("Average Energy Error",size = 15)
# ax1.legend(labels, loc = (0, -0.13),fontsize = 12,ncol=3)
# #ax1.set_yscale('log')
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
# ax1.scatter(aListSR, avStateErrSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(aListVar, avStateErrVar, color = colors[1], label=labels[1], marker = '>')
# #ax1.scatter(NListHeiF, avStateErrHeiF, color = colors[2], label=labels[2], marker = '<')
# ax1.set_xlabel("Alpha",size = 15)
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
# ax1.scatter(aListSR, runsOverSR, color = colors[0], label=labels[0], marker = '^')
# ax1.scatter(aListVar, runsOverVar, color = colors[1], label=labels[1], marker = '>')
# #ax1.scatter(NListHeiF, runsOverHeiF, color = colors[2], label=labels[2], marker = '<')
# ax1.set_xlabel("Alpha",size = 15)
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


#
#
# fit = np.polyfit(np.log(NListSR), np.log(avRunTimeSR), 1, full=True)
# fitPoints = (np.e**fit[0][1])*NListSR**fit[0][0]
# print('fit: ', fit)
# plt.figure(figsize=(8,8))
# labels = ['Central Spin with Constant A','Heisenberg', 'Central Spin with Variable A']
# colors = ['blue', 'green', 'red']
# ttl = plt.suptitle("Average Run Time Scaling \n"+r"Central Spin with Constant A" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.97])
# ax1 = plt.subplot(gs[0, 0])
# #ax1.set_ylim(-0.2,1)
# ax1.scatter(NListSR, avRunTimeSR, color = colors[0], label=labels[0], marker = '^')
# ax1.plot(NListSR, fitPoints, color = colors[0], linestyle = '--')
# #ax1.scatter(NListHei,avRunTimeHei, color = colors[1], label=labels[1])
# #ax1.scatter(NListVar, avRunTimeVar, color = colors[2], label=labels[2], marker = '>')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Run Time (s)",size = 15)
# #ax1.legend(labels, loc = (-0.1, -0.13),fontsize = 12,ncol=3)
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# plt.show()
#
# print('c= ', np.e**fit[0][1])
# print('m= ', fit[0][0])