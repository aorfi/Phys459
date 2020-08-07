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
    engErrNKAll.append(engErrNK)
    engErrSRAll.append(engErrSR)
    stateErrAll.append(stateErr)
    stateErrNKAll.append(stateErrNK)
    stateErrSRAll.append(stateErrSR)
    runTimeAll.append(runTime)
    runTimeNKAll.append(runTimeNK)
    runTimeSRAll.append(runTimeSR)

NListNK = np.arange(6, 12)
for i in range(len(NListNK)):
    N = NListNK[i]
    M = N
    dataLocation = "Data/08-07-20/nkN" + str(N) + "M" + str(M) + "B1.json"
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

AList = np.arange(1, 5)
for i in range(len(AList)):
    N = 4
    M = AList[i]*N
    dataLocation = "Data/08-07-20/nkN" + str(N) + "M" + str(M) + "B1.json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    engErrNK, engErrSR, stateErrNK, stateErrSR,runTimeNK, runTimeSR = saved
    AengErrNK.append(engErrNK)
    AstateErrNK.append(stateErrNK)
    ArunTimeNK.append(runTimeNK)
    AengErrSR.append(engErrSR)
    AstateErrSR.append(stateErrSR)
    ArunTimeSR.append(runTimeSR)




#
# # ***** Histogram ****
# index = 3
# N=4
# M=N*(index+1)
# B=1
# #index = N-2
# # Plotting
# # allEngErr = [engErrNKAll[index],engErrSRAll[index], engErrAll[index]]
# # allStateErr = [stateErrNKAll[index],stateErrSRAll[index], stateErrAll[index]]
# # allRunTime = [ runTimeNKAll[index], runTimeSRAll[index], runTimeAll[index]]
# # labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration', 'Non-NetKet RBM']
# # colors = ['blue', 'green', 'red']
# # allEngErr = [engErrNKAll[index],engErrSRAll[index]]
# # allStateErr = [stateErrNKAll[index],stateErrSRAll[index]]
# # allRunTime = [ runTimeNKAll[index], runTimeSRAll[index]]
# # labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration']
# # colors = ['blue', 'green']
# allEngErr = [AengErrNK[index],AengErrSR[index]]
# allStateErr = [AstateErrNK[index],AstateErrSR[index]]
# allRunTime = [ ArunTimeNK[index], ArunTimeSR[index]]
# labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration']
# colors = ['blue', 'green']
#
#
# hisIt= np.arange(50)
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# ttl = plt.suptitle("Comparison of NetKet and Non-NetKet RBM \n N = " + str(N)+", B = "+str(B)+", M = " + str(M),size =20)
# gs = gridspec.GridSpec(ncols=3, nrows=3, hspace = 0.4)
# ttl.set_position([.5, 0.94])
# ax1 = plt.subplot(gs[0, 0])
# ax1.hist(allEngErr, bins=10, color = colors, label=labels)
# ax1.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
# ax2 = plt.subplot(gs[0, 1])
# ax2.hist(allStateErr, bins=10, color = colors, label=labels)
# ax2.set_xlabel("$1-|<\Psi_{RBM}|\Psi_{ED}>|^2$",size = 15)
#
# ax3 = plt.subplot(gs[0, 2])
# ax3.hist(allRunTime, bins=10, color = colors)
# ax3.set_xlabel("Runtime (s)",size = 15)
#
# ax4 = plt.subplot(gs[1, :])
# # ax4.scatter(hisIt,engErrNKAll[index], color = 'blue')
# # ax4.scatter(hisIt,engErrSRAll[index], color = 'green',marker = '>')
# #ax4.scatter(hisIt,engErrAll[index], color = 'red', marker = '^')
# ax4.scatter(hisIt,AengErrNK[index], color = 'blue')
# ax4.scatter(hisIt,AengErrSR[index], color = 'green',marker = '>')
# ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)
#
# ax1.legend(labels, loc = (0, -3.3),fontsize = 12,ncol=3)
#
# ax5 = plt.subplot(gs[2, :])
# #ax5.set_yscale('log')
# # ax5.scatter(hisIt,runTimeNKAll[index], color = 'blue')
# # ax5.scatter(hisIt,runTimeSRAll[index], color = 'green',marker = '>')
# #ax5.scatter(hisIt,runTimeAll[index], color = 'red', marker = '^')
# ax5.scatter(hisIt,ArunTimeNK[index], color = 'blue')
# ax5.scatter(hisIt,ArunTimeSR[index], color = 'green',marker = '>')
# ax5.set_xlabel("Run Number",size = 15)
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
#for changing alpha
AavEngErrNK = []
AavEngErrSR = []
AavStateErrNK = []
AavStateErrSR = []
AavRunTimeNK = []
AavRunTimeSR = []
ArunsOverNK = []
ArunsOverSR = []
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
for i in range(len(AengErrNK)):
    AavEngErrNKTemp = np.sum(AengErrNK[i]) / (50)
    AavEngErrNK.append(AavEngErrNKTemp)
    AavEngErrSRTemp = np.sum(AengErrSR[i]) / (50)
    AavEngErrSR.append(AavEngErrSRTemp)
    AavStateErrNKTemp = np.sum(AstateErrNK[i]) / (50)
    AavStateErrNK.append(AavStateErrNKTemp)
    AavStateErrSRTemp = np.sum(AstateErrSR[i]) / (50)
    AavStateErrSR.append(AavStateErrSRTemp)
    AavRunTimeNKTemp = np.sum(ArunTimeNK[i]) / (50)
    AavRunTimeNK.append(AavRunTimeNKTemp)
    AavRunTimeSRTemp = np.sum(ArunTimeSR[i]) / (50)
    AavRunTimeSR.append(AavRunTimeSRTemp)
    ArunsOverNKTemp = sum(j > cutOff for j in AengErrNK[i])
    ArunsOverNK.append(ArunsOverNKTemp)
    ArunsOverSRTemp = sum(j > cutOff for j in AengErrSR[i])
    ArunsOverSR.append(ArunsOverSRTemp)

#
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
# ax1.set_ylabel("Average Runtime",size = 15)
# ax1.legend(labels, loc = (0, 1),fontsize = 12,ncol=3)
# #plt.show()
#
# # ***** Energy Error Scaling ******
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

# ***** Number  of runs******
NRange= np.arange(2,len(avRunTime)+2)
NRangeNK= np.arange(2,len(avRunTimeNK)+2)
labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration', 'Non-NetKet RBM']
colors = ['blue', 'green', 'red']
#plt.figure(constrained_layout=True)
plt.figure(figsize=(10,10))
ttl = plt.suptitle("Number of Runs with Energy Error above "+str(cutOff) +r" $\alpha = 1$" ,size =20)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.94])
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(NRangeNK, runsOverNK, color = colors[0], label=labels[0])
ax1.scatter(NRangeNK, runsOverSR, color = colors[1], label=labels[1])
ax1.scatter(NRange, runsOver, color = colors[2], label=labels[2])
ax1.set_xlabel("N",size = 15)
ax1.set_ylabel("Number of Runs",size = 15)
ax1.legend(labels, loc = (0, 1),fontsize = 12,ncol=3)
plt.show()


# #******* Changing Alpha *****
# Alphas= np.arange(1,5)
# labels = ['NetKet Gradient Descent','NetKet Stochastic Reconfiguration']
# colors = ['blue', 'green']
# #plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# ttl = plt.suptitle("RunTime"+ " for N=4"+r" $\alpha = 1-3$",size =20)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.94])
# ax1 = plt.subplot(gs[0, 0])
# ax1.scatter(Alphas, AavRunTimeNK, color = colors[0], label=labels[0])
# ax1.scatter(Alphas,AavRunTimeSR, color = colors[1], label=labels[1])
# ax1.set_xlabel(r"$\alpha$",size = 15)
# ax1.set_ylabel("RunTime",size = 15)
# ax1.legend(labels, loc = (0, 1),fontsize = 12,ncol=3)
# plt.show()

print(1-0.5**5)