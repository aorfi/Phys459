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
# engErr = []
# stateErr = []
# time = []
# lengths = []
# for i in range(4):
#     for j in range(10):
#         N = i + 2
#         M = j + 1
#         print('N,M', N,M)
#         dataLocation = 'Data/06-08-20/OneRunN' + str(N) + 'M' + str(M) + '.json'
#         saved = []
#         with open(dataLocation) as file:
#             for line in file:
#                 saved.append(json.loads(line))
#
#         cgdTime, cgdEngErr, cgdStateErr, edTime, length = saved
#         engErr.append(cgdEngErr)
#         stateErr.append(cgdStateErr)
#         time.append(cgdTime)
#         lengths.append(length)

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
#
# avEngErr = []
# avStateErr = []
# avRunTime = []
# numFailure = []
# numFailureErr = []
# avEngErrF = []
# avRunTimeF = []
# numFailureF = []
# numFailureErrF = []
# cutOff = 0.1
#
# for i in range(len(engErr)):
#     avEngErrTemp = np.sum(engErr[i]) / (lengths[i])
#     avEngErr.append(avEngErrTemp)
#     avStateErrTemp = np.sum(stateErr[i]) / (lengths[i])
#     avStateErr.append(avStateErrTemp)
#     avRunTimeTemp = np.sum(time[i]) / (lengths[i])
#     avRunTime.append(avRunTimeTemp)
#
#     runsCutOff = sum(i > cutOff for i in engErr[i])
#     numFailure.append(runsCutOff)
#     numFailureErr.append(runsCutOff ** (0.5))

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



# ****** Import NetKet Data ******
engErrAll = []
engErrNKAll = []
engErrSRAll = []
stateErrAll = []
stateErrNKAll = []
stateErrSRAll = []
runTimeAll = []
runTimeNKAll = []
runTimeSRAll = []
#For changing alpha
AengErrNK = []
AstateErrNK = []
ArunTimeNK = []
AengErrSR = []
AstateErrSR = []
ArunTimeSR = []

NList = np.arange(2,6)
for i in range(len(NList)):
    N = NList[i]
    M=N
    dataLocation = "Data/08-07-20/N"+str(N)+"M"+str(M)+"B1.json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErrNK,engErrSR, engErr, stateErrNK, stateErrSR, stateErr, runTimeNK,runTimeSR, runTime= saved
    engErrAll.append(engErr)
    stateErrAll.append(stateErr)
    runTimeAll.append(runTime)


NListNK = np.arange(2, 12)
for i in range(len(NListNK)):
    N = NListNK[i]
    M = N
    dataLocation = "Data/08-14-20/nkN" + str(N) + "M" + str(M) + "B1.json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErrNK, engErrSR, stateErrNK, stateErrSR,runTimeNK, runTimeSR = saved
    engErrNKAll.append(engErrNK)
    engErrSRAll.append(engErrSR)
    stateErrNKAll.append(stateErrNK)
    stateErrSRAll.append(stateErrSR)
    runTimeNKAll.append(runTimeNK)
    runTimeSRAll.append(runTimeSR)


#
#***** Histogram ****

N=6
M=N
B=1
index = N-2
# Plotting
# allEngErr = [engErrNKAll[index],engErrSRAll[index], engErrAll[index]]
# allStateErr = [stateErrNKAll[index],stateErrSRAll[index], stateErrAll[index]]
# allRunTime = [ runTimeNKAll[index], runTimeSRAll[index], runTimeAll[index]]
labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration', 'Non-NetKet RBM']
colors = ['blue', 'green', 'red']
# allEngErr = [engErrNKAll[index],engErrSRAll[index]]
# allStateErr = [stateErrNKAll[index],stateErrSRAll[index]]
# allRunTime = [ runTimeNKAll[index], runTimeSRAll[index]]
# labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration']
# colors = ['blue', 'green']

#
# hisIt= np.arange(50)
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# ttl = plt.suptitle("NetKet Stochastic Reconfiguration \n N = " + str(N)+", M = " + str(M)+", B = "+str(B)+", A = 1",size =20)
# gs = gridspec.GridSpec(ncols=3, nrows=3, hspace = 0.4)
# ttl.set_position([.5, 0.96])
#
# ax1 = plt.subplot(gs[0, 0])
# ax1.hist(engErrSRAll[index], color= 'g')
# #ax1.hist(allEngErr, bins=10, color = colors, label=labels)
# ax1.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
# ax2 = plt.subplot(gs[0, 1])
# ax2.hist(stateErrSRAll[index], color= 'g')
# #ax2.hist(allStateErr, bins=10, color = colors, label=labels)
# ax2.set_xlabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$",size = 15)
#
# ax3 = plt.subplot(gs[0, 2])
# ax3.hist(runTimeSRAll[index], color= 'g')
# #ax3.hist(allRunTime, bins=10, color = colors)
# ax3.set_xlabel("Runtime (s)",size = 15)
#
# ax4 = plt.subplot(gs[1, :])
# #ax4.scatter(hisIt,engErrNKAll[index], color = 'blue')
# ax4.scatter(hisIt,engErrSRAll[index], color = 'green',marker = '>')
# #ax4.scatter(hisIt,engErrAll[index], color = 'red', marker = '^')
# ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)
#
# #ax1.legend(labels, loc = (-0.3, -3.3),fontsize = 12,ncol=3)
#
# ax5 = plt.subplot(gs[2, :])
# #ax5.set_yscale('log')
# #ax5.scatter(hisIt,runTimeNKAll[index], color = 'blue')
# ax5.scatter(hisIt,runTimeSRAll[index], color = 'green',marker = '>')
# #ax5.scatter(hisIt,runTimeAll[index], color = 'red', marker = '^')
# ax4.set_xlabel("Run Number",size = 15)
# ax5 .set_ylabel("Runtime (s)", size = 15)
# plt.show()

# **** Get averages *****
avEngErr = []
avEngErrNK = []
avEngErrSR = []
avStateErr = []
avStateErrNK = []
avStateErrSR = []
avRunTime = []
avRunTimeNK = []
avRunTimeSR = []
cutOff = 0.01 #### *******
runsOver = []
runsOverNK = []
runsOverSR = []

for i in range(len(engErrAll)):
    avEngErrTemp = np.sum(engErrAll[i]) / (50)
    avEngErr.append(avEngErrTemp)
    avStateErrTemp = np.sum(stateErrAll[i]) / (50)
    avStateErr.append(avStateErrTemp)
    avRunTimeTemp = np.sum(runTimeAll[i]) / (50)
    avRunTime.append(avRunTimeTemp)
    runsOverTemp = sum(j > cutOff for j in engErrAll[i])
    runsOver.append(runsOverTemp)
for i in range(len(engErrNKAll)):
    avEngErrNKTemp = np.sum(engErrNKAll[i]) / (50)
    avEngErrNK.append(avEngErrNKTemp)
    avEngErrSRTemp = np.sum(engErrSRAll[i]) / (50)
    avEngErrSR.append(avEngErrSRTemp)
    avStateErrNKTemp = np.sum(stateErrNKAll[i]) / (50)
    avStateErrNK.append(avStateErrNKTemp)
    avStateErrSRTemp = np.sum(stateErrSRAll[i]) / (50)
    avStateErrSR.append(avStateErrSRTemp)
    avRunTimeNKTemp = np.sum(runTimeNKAll[i]) / (50)
    avRunTimeNK.append(avRunTimeNKTemp)
    avRunTimeSRTemp = np.sum(runTimeSRAll[i]) / (50)
    avRunTimeSR.append(avRunTimeSRTemp)
    runsOverNKTemp = sum(j > cutOff for j in engErrNKAll[i])
    runsOverNK.append(runsOverNKTemp)
    runsOverSRTemp = sum(j > cutOff for j in engErrSRAll[i])
    runsOverSR.append(runsOverSRTemp)

#***** Run Time Scaling ******


NRangeNK= np.arange(2,len(avRunTimeNK)+2)
fit = np.polyfit(np.log(NRangeNK), np.log(avRunTimeSR), 1)
fitPoints = (np.e**fit[1])*NRangeNK**fit[0]
print('fit: ', fit)
#plt.figure(constrained_layout=True)
plt.figure(figsize=(6,6))
ttl = plt.suptitle("Runtime Scaling \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =18)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.98])
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(NRangeNK, avRunTimeSR)
ax1.plot(NRangeNK, fitPoints)
ax1.set_xlabel("N",size = 15)
ax1.set_ylabel("Average Runtime (s)",size = 15)
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.show()



# # ***** Energy Error Scaling ******
#plt.figure(constrained_layout=True)
plt.figure(figsize=(8,8))
ttl = plt.suptitle("Energy Error Scaling \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.97])
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(NRangeNK, avEngErrSR)
ax1.set_xlabel("N",size = 15)
ax1.set_ylabel("Average Energy Error",size = 15)
#ax1.set_yscale('log')
plt.show()

# # ***** State Error Scaling ******
#plt.figure(constrained_layout=True)
plt.figure(figsize=(8,8))
ttl = plt.suptitle("State Error Scaling \n"+r"Stochastic Reconfiguration with $\alpha = 1$" ,size =20)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.97])
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(NRangeNK, avStateErrSR)
ax1.set_xlabel("N",size = 15)
ax1.set_ylabel("Average State Error",size = 15)
#ax1.set_yscale('log')
plt.show()
#
# # # ***** Number  of runs******
#plt.figure(constrained_layout=True)
plt.figure(figsize=(8,8))
ttl = plt.suptitle("Number of Runs with Energy Error above "+str(cutOff) +r" $\alpha = 1$" ,size =20)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.97])
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(NRangeNK, runsOverSR)
ax1.set_xlabel("N",size = 15)
ax1.set_ylabel("Number of Runs",size = 15)
plt.show()




# #
# # ***** Run Time Scaling ******
# NRange= np.arange(2,len(avRunTime)+2)
# NRangeNK= np.arange(2,len(avRunTimeNK)+2)
# labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration', 'Non-NetKet RBM']
# colors = ['blue', 'green', 'red']
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# ttl = plt.suptitle("Runtime Scaling "+r"$\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.94])
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(NRangeNK, avRunTimeNK, color = colors[0], label=labels[0])
# ax1.scatter(NRangeNK, avRunTimeSR, color = colors[1], label=labels[1])
# ax1.scatter(NRange, avRunTime, color = colors[2], label=labels[2])
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Runtime (s)",size = 15)
# ax1.legend(labels, loc = (0, 1),fontsize = 12,ncol=3)
# ax1.set_yscale('log')
# #plt.show()
# #
# # # ***** Energy Error Scaling ******
# NRange= np.arange(2,len(avRunTime)+2)
# NRangeNK= np.arange(2,len(avRunTimeNK)+2)
# labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration', 'Non-NetKet RBM']
# colors = ['blue', 'green', 'red']
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# ttl = plt.suptitle("Energy Error Scaling "+r"$\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.94])
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(NRangeNK, avEngErrNK, color = colors[0], label=labels[0])
# ax1.scatter(NRangeNK, avEngErrSR, color = colors[1], label=labels[1])
# ax1.scatter(NRange, avEngErr, color = colors[2], label=labels[2])
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Energy Error",size = 15)
# ax1.legend(labels, loc = (0, 1),fontsize = 12,ncol=3)
# #ax1.set_yscale('log')
# #plt.show()
#
# # # ***** Number  of runs******
# NRange= np.arange(2,len(avRunTime)+2)
# NRangeNK= np.arange(2,len(avRunTimeNK)+2)
# labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration', 'Non-NetKet RBM']
# colors = ['blue', 'green', 'red']
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# ttl = plt.suptitle("Number of Runs with Energy Error above "+str(cutOff) +r" $\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.94])
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(NRangeNK, runsOverNK, color = colors[0], label=labels[0])
# ax1.scatter(NRangeNK, runsOverSR, color = colors[1], label=labels[1])
# ax1.scatter(NRange, runsOver, color = colors[2], label=labels[2])
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Number of Runs",size = 15)
# ax1.legend(labels, loc = (0, 1),fontsize = 12,ncol=3)
# #plt.show()
#
# # Covert Number of failed runs into probability of failure
# probSuc = 0.90
# probFailSR = []
# numRuns = []
# for i in runsOverSR:
#     probFailSR.append(i/50)
# # Find the runs needed
# for i in probFailSR:
#     if i == 0 :
#         runs = 1
#     else:
#         runs = np.log(1-probSuc)/np.log(i)
#         runs = int(runs)+1
#     numRuns.append(runs)
# # Scale the runtime
# scaledRunTimeSR = []
# for i in range(len(numRuns)):
#     scaledRunTimeSR.append(numRuns[i]*avRunTimeSR[i])
#
# # ***** Run Time Scaling ******
# NRange= np.arange(2,len(avRunTime)+2)
# NRangeNK= np.arange(2,len(avRunTimeNK)+2)
# labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration', 'Non-NetKet RBM']
# colors = ['blue', 'green', 'red']
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# ttl = plt.suptitle("Scaled Runtime to have  a 90% Chance of Success \n Stochastic Reconfiguration "+r"$\alpha = 1$" ,size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.94])
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(NRangeNK, scaledRunTimeSR, color = 'green')
# ax1.set_xlabel("N",size = 15)
# ax1.set_ylabel("Average Runtime (s)",size = 15)
# ax1.set_yscale('log')
# plt.show()
#
# print(numRuns)
#
#
#
#
