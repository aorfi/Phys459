import netket as nk
import json
import os
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
import time
from pickle import load, dump
from collections import OrderedDict
import multiprocess as mp
#from multiprocess.pool import ThreadPool as Pool
import os
plt.style.use('seaborn')



#Central Spin Hamiltonian and Hilbert space defined in NetKet objects
def CSHam(N, B, A):
    # Make graph with no edges of length N
    #g = nk.graph.Edgeless(N)
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    # Define sigma matrices
    sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
    sigmax = 0.5 * np.array([[0, 1], [1, 0]])
    sigmay = 0.5 * np.array([[0, -1j], [1j, 0]])
    operators = []
    sites = []
    # Central spin term
    operators.append((B * sigmaz).tolist())
    sites.append([0])
    # Iteraction term
    itOp = np.kron(sigmaz, sigmaz) + np.kron(sigmax, sigmax) + np.kron(sigmay, sigmay)
    for i in range(N - 1):
        operators.append((A * itOp).tolist())
        sites.append([0, (i+1)])
    ha = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    #Returns Hamiltonian and Hilbert space
    return ha, hi

def CSVarAHam(N, B, A, N0):
    # Make graph with no edges of length N
    #g = nk.graph.Edgeless(N)
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    # Define sigma matrices
    sigmaz = -0.5 * np.array([[1, 0], [0, -1]])
    sigmax = 0.5 * np.array([[0, 1], [1, 0]])
    sigmay = -0.5 * np.array([[0, -1j], [1j, 0]])
    operators = []
    sites = []
    # Central spin term
    operators.append((B * sigmaz).tolist())
    sites.append([0])
    # Iteraction term
    itOp = np.kron(sigmaz, sigmaz) + np.kron(sigmax, sigmax) + np.kron(sigmay, sigmay)
    for i in range(N - 1):
        Ak = A / (N0) * np.exp(-i / N0)
        operators.append((Ak * itOp).tolist())
        sites.append([0, (i+1)])
    ha = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    #Returns Hamiltonian and Hilbert space
    return ha, hi

# Ferromagnetic Heisenberg Hamtilonian with Field
def heiHam(N, J, h):
    # Make graph with no edges of length N
    #g = nk.graph.Edgeless(N)
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    # Define sigma matrices
    sigmaz = 0.5 * np.array([[1, 0], [0, -1]])
    sigmax = 0.5 * np.array([[0, 1], [1, 0]])
    sigmay = 0.5 * np.array([[0, -1j], [1j, 0]])
    operators = []
    sites = []
    # Iteraction term
    itOp = J*np.kron(sigmaz, sigmaz) + J*np.kron(sigmax, sigmax) + J*np.kron(sigmay, sigmay)
    for i in range(N-1):
        operators.append((itOp).tolist())
        sites.append([i, (i+1)])
    #Field Term
    for i in range(N):
        operators.append((h*sigmaz).tolist())
        sites.append([i])
    ha = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    #Returns Hamiltonian and Hilbert space
    return ha, hi

# Exact Digonalization NetKet
def exactDigonalization(ha):
    haMatrix = ha.to_dense()
    e, v = np.linalg.eigh(haMatrix)
    idx = e.argsort()[::1]
    values = e[idx]
    vectors = v[:, idx].T
    return values, vectors

# NetKet RBM with stocastic reconfiguration descent
class RBM:
    def __init__(self, N, ha, hi, ma):
        self.ha, self.hi, self.ma = ha, hi, ma
        # Define sampler
        self.sa = nk.sampler.MetropolisLocal(machine=self.ma)
        # Optimizer
        self.op = nk.optimizer.Sgd(learning_rate=0.05)
        self.N = N

    def __call__(self, output):
        N = self.N
        gs = nk.variational.Vmc(hamiltonian=self.ha,
                                sampler=self.sa,
                                optimizer=self.op,
                                n_samples=1000,
                                use_iterative=True,
                                method='Sr')
        start = time.time()
        gs.run(output_prefix=output + 'SR', n_iter=1000)
        end = time.time()
        runTime = end - start
        # import the data from log file
        data = json.load(open(output + "SR.log"))
        # Extract the relevant information
        iters = []
        energy_RBM = []

        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            engTemp = iteration["Energy"]["Mean"]
            energy_RBM.append(engTemp)
        finalEng = energy_RBM[-1]
        maArray = self.ma.to_array()
        # finalState = 0
        # for i in range(2 ** N):
        #     finalState += maArray[2 ** N - 1 - i] * basis[0][i]
        return finalEng, maArray, runTime

# get randomized RBM parameters (between zero and 1)
def ranRBMpar(N, alpha):
    M = alpha * N
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    par = 1 - 2 * np.random.rand(2 * (N + M + N * M))
    return par

# Initialization of NetKet Parameters (Change this to NETKET)
def covertParams(N,alpha,par, ma):
    # Change to a,b,w
    M = alpha * N
    num = N + M + N * M
    parC = np.vectorize(complex)(par[:num], par[num:])
    a = parC[:N]
    a = [0.5 * x for x in a]
    b = parC[N:N + M]
    w = parC[N + M:].reshape(M, N)
    w = [0.5 * x for x in w]
    w = np.array(w).T
    rbmOrderedDict = OrderedDict([('a', a), ('b', b), ('w', w)])
    # Save parameters so they can be loaded into the netket machine
    with open("Logs/par"+str(par[0])+".json", "wb") as output:
        dump(rbmOrderedDict, output)
    # Load into ma
    ma.load("Logs/par"+str(par[0])+".json")

# Error Calculation
def err(state, edState, eng, edEng):
    engErr = np.abs(eng - edEng)
    overlap = np.dot(state.conj().T,edState)
    waveFunctionErr = 1 - np.dot(overlap.conj().T,overlap).real
    return engErr, waveFunctionErr

def runDescentCS(N,B,A,alpha):
    ha, hi = CSHam(N,B,A)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    ma.init_random_parameters(1)
    rbm = RBM(N, ha, hi, ma)
    eng, state, runTime = rbm("Logs/CS"+str(N))
    return eng, state, runTime

def runDescentCSVarA(N, B, A, N0,alpha):
    ha, hi = CSVarAHam(N,B,A,N0)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    ma.init_random_parameters(1)
    rbm = RBM(N, ha, hi, ma)
    eng, state, runTime = rbm("Logs/CSVarA"+str(N))
    return eng, state, runTime

def runDescentHei(N,J,h,alpha):
    ha, hi = heiHam(N,J,h)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    ma.init_random_parameters(1)
    rbm = RBM(N, ha, hi, ma)
    eng, state, runTime = rbm("Logs/Hei"+str(N))
    return eng, state, runTime

#Test
alpha = 1
B=1
A=1
J = 1
h=1
N = 10
N0 = N / 2
M = alpha*N
# ha, hi = CSVarAHam(N,B,A,N0)
# ha, hi = CSHam(N,B,A)
ha, hi = heiHam(N,J,h)
print(exactDigonalization(ha)[0])

#
#
#
# # Running information
#
# # Hamiltionian Parameters
# alpha = 1
# B=1
# A=1
# J = 1
# h=1
# NList = np.arange(2,11)
#
#
# for i in range(len(NList)):
#     N = NList[i]
#     N0 = N / 2
#     M = alpha*N
#     ha, hi = CSHam(N,B,A)
#
#     # ha, hi = CSVarAHam(N, B, A,N0)
#
#     # # Exact Diagonalization
#     e,v = exactDigonalization(ha)
#     edEng = e[0]
#     print("ed Energy: ", edEng)
#     edState = v[0]
#     print("ed State: ", edState)
#
#     # # Histogram All
#     hisIt = np.arange(50)
#     engErr = []
#     stateErr = []
#     runTime = []
#
#     # Node Information
#     ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=50))
#     pool = mp.Pool(processes=ncpus)
#
#     resultsSR = [pool.apply(runDescentCS, args = (N,B,A,alpha)) for x in hisIt]
#     # resultsSR = [pool.apply(runDescentHei, args=(N, J, h, alpha)) for x in hisIt]
#     # resultsSR = [pool.apply(runDescentCSVarA, args=(N,B,A,N0,alpha)) for x in hisIt]
#
#     for i in range(len(hisIt)):
#         # NK SR Run
#         engTemp, stateTemp, runTimeTemp = resultsSR[i]
#         # print('RBM Eng: ', engTemp)
#         # print('RBM State: ', stateTemp)
#         runTime.append(runTimeTemp)
#         errSR = err(stateTemp, edState, engTemp, edEng)
#         engErr.append(errSR[0])
#         stateErr.append(errSR[1])
#     print('State Error: ', stateErr)
#     print('Eng Error: ', engErr)
#
#     #Save data to JSON file
#     data = [engErr, stateErr, runTime]
#     fileName = "Data/10-27-20/csN" + str(N) + "M" + str(M) + ".json"
#     # fileName = "Data/10-27-20/heiFN" + str(N) + "M" + str(M) + ".json"
#     # fileName = "Data/10-27-20/csVarAN"+str(N)+"M" + str(M)+".json"
#     open(fileName, "w").close()
#     with open(fileName, 'a') as file:
#         for item in data:
#             line = json.dumps(item)
#             file.write(line + '\n')
#     print('SAVED')