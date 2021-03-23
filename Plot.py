import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('seaborn')


# ****** Import Timing Data ******
fb = []
samples = []
eng = []
dp = []
NList = np.arange(2, 13)
for i in range(len(NList)):
    N = NList[i]
    M = N
    dataLocation = "Data/21-03-23/timeN"+ str(N) + "M" + str(M) + ".json"
    saved = []
    with open(dataLocation) as file:
        for line in file:
            saved.append(json.loads(line))
    [fbTime, samplesTime, engTime, dpTime]= saved
    fbTime.pop(0)
    samplesTime.pop(0)
    engTime.pop(0)
    dpTime.pop(0)
    fb.append(fbTime)
    samples.append(samplesTime)
    eng.append(engTime)
    dp.append(dpTime)

runtime = []
fbAv = []
samplesAv = []
engAv = []
dpAv = []
for i in range (len(fb)):
    runtime.append(sum(fb[i]))
    fbAv.append(sum(fb[i])/len(fb[i]))
    samplesAv.append(sum(samples[i]) / len(samples[i]))
    engAv.append(sum(eng[i]) / len(eng[i]))
    dpAv.append(sum(dp[i]) / len(dp[i]))

#
plt.figure(figsize=(8, 10))
plt.title("Runtime Breakdown", size=20)
labels = ['Samples','Energy Estimate', 'Update Params']

# #plt.scatter(NList, runtime)
# plt.scatter(NList, samplesAv, label = labels[0])
# plt.scatter(NList, engAv, label = labels[1])
# plt.scatter(NList,dpAv, label = labels[2])
# plt.ylabel("Runtime (s)", size=15)
# plt.xlabel("N", size=15)
# plt.legend(loc = (-0.1, -0.15),fontsize = 12,ncol=4)
# plt.show()


#
index = 4
N=index+2
times = [samples[index], eng[index], dp[index]]
plt.figure(figsize=(8, 10))
plt.title("Runtime N = "+str(N), size=20)
labels = ['Samples','Energy Estimate', 'Update Params']

plt.hist(times, density=False, bins=20,histtype='bar', range=[0, 0.08], ec='black', label = labels)
plt.ylabel("Counts", size=15)
plt.xlabel("Runtime (s)", size=15)
plt.legend(loc = (-0.1, -0.15),fontsize = 12,ncol=4)
plt.show()

#