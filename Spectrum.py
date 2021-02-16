# Activate the qutip-env-linux environment (source qutip-env-linux/bin/activate)
# If running on the cluster then you also need mpi4py which is preloaded on Beluga (module load mpi4py)
import netket as nk
import json
from qutip import *
import numpy as np
import time
import multiprocess as mp
from collections import OrderedDict
from pickle import dump
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('seaborn')
from scipy.stats import norm


# Central Spin Hamiltonian and Hilbert space, inputs are Hamiltonian parameters
def CSHam(N, B, Ak):
    # Make graph with of length N with no periodic boundary conditions
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hilbertSpace = nk.hilbert.Spin(s=0.5, graph=g)
    # Define pauli matrices
    pauliz = 0.5 * np.array([[1, 0], [0, -1]])
    paulix = 0.5 * np.array([[0, 1], [1, 0]])
    pauliy = 0.5 * np.array([[0, -1j], [1j, 0]])
    operators = []
    sites = []
    # Central spin term
    operators.append((B * pauliz).tolist())
    sites.append([0])
    # Interaction term
    itOp = np.kron(pauliz, pauliz) + np.kron(paulix, paulix) + np.kron(pauliy, pauliy)
    for i in range(N - 1):
        operators.append((Ak[i] * itOp).tolist())
        sites.append([0, (i+1)])
    # Create hamiltonian
    hamiltonian = nk.operator.LocalOperator(hilbertSpace, operators=operators, acting_on=sites)
    return hamiltonian, hilbertSpace

# Heisenberg  Hamiltonian and Hilbert space, inputs are Hamiltonian parameters
def heiHam(N, J, h):
    # Make graph with of length N with no periodic boundary conditions
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hilbertSpace = nk.hilbert.Spin(s=0.5, graph=g)
    # Define pauli matrices
    pauliz = 0.5 * np.array([[1, 0], [0, -1]])
    paulix = 0.5 * np.array([[0, 1], [1, 0]])
    pauliy = 0.5 * np.array([[0, -1j], [1j, 0]])
    operators = []
    sites = []
    # Interaction term
    itOp = J*np.kron(pauliz, pauliz) + J*np.kron(paulix, paulix) + J*np.kron(pauliy, pauliy)
    for i in range(N-1):
        operators.append((itOp).tolist())
        sites.append([i, (i+1)])
    # Field Term
    for i in range(N):
        operators.append((h*sigmaz).tolist())
        sites.append([i])
    hamiltonian = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    return hamiltonian, hilbertSpace

# Exact Diagonalization, returns eigenvalues and vectors, inputs Hamiltonian
def exactDiagonalization(hamiltonian):
    # Changes Hamiltonian to matrix form
    haMatrix = hamiltonian.to_dense()
    # Gets eigenvalues and vectors
    e, v = np.linalg.eigh(haMatrix)
    # Orders from smallest to largest
    idx = e.argsort()[::1]
    eigenValues = e[idx]
    eigenVectors = v[:, idx].T
    return eigenValues, eigenVectors

# NetKet RBM with stochastic reconfiguration descent
class RBM:
    def __init__(self, N, hamiltonian, hilbertSpace, machine):
        # Assign inputs
        self.hamiltonian, self.hilbertSpace, self.machine, self.N = hamiltonian, hilbertSpace, machine, N
        # Define sampler
        self.sampler = nk.sampler.MetropolisLocal(machine=self.machine)
        # Define optimizer
        self.optimizer = nk.optimizer.Sgd(self.machine,learning_rate=0.05)

    # Output is the name of the output file in which the descent data is stored
    def __call__(self, output):
        gs = nk.Vmc(hamiltonian=self.hamiltonian,
                                sampler=self.sampler,
                                optimizer=self.optimizer,
                                # Number of samples use in the MC estimation
                                n_samples=1000,
                                sr = nk.optimizer.SR(self.machine))
        # Start timing
        start = time.time()
        # Set the output files as well as number of iterations in the descent
        gs.run(out=output, n_iter=1000)
        end = time.time()
        runTime = end - start
        # Import the data from log file
        data = json.load(open(output+'.log'))
        # Extract the relevant information
        iters = []
        energy_RBM = []
        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            engTemp = iteration["Energy"]["Mean"]
            energy_RBM.append(engTemp)
        finalEng = energy_RBM[-1]
        # Get machine statethe state of the machine as an array
        state = self.machine.to_array()
        # Outputs the final energy, the final state, and the runtime
        return finalEng, state, runTime

# Error Calculation (Input: the found state, the state from exact diagonalization, the found energy, the energy from exact diagonalization)
def err(state, edState, eng, edEng,N):
    engErr = np.abs(eng - edEng)
    overlap = np.dot(state.conj().reshape(2**N, 1).T, edState.reshape(2**N, 1))
    waveFunctionErr = 1 - np.linalg.norm(overlap)
    return engErr, waveFunctionErr

# Combines all steps into a function to run on the cluster
def runDescentCS(N,B,Ak,alpha):
    # Define hamiltonian and hibert space (need to do this here cause can't use netket objects as input to use multiprocessing functions)
    ha, hi = CSHam(N,B,Ak)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    # Initialize the RBM parameters (Gaussian with sd 0.25)
    ma.init_random_parameters(seed=None, sigma=0.25)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma)
    # Run RBM
    eng, state, runTime = rbm("Logs/CS"+str(N))
    return eng, state, runTime

def parameterOutputList(machine):
    paramsComplex = machine.parameters
    parameters = []
    for i in paramsComplex:
        parameters.append(i.real)
        parameters.append(i.imag)
    return parameters

# Initializes random RBM parameters
def ranPar(N, alpha, ma):
    M = alpha * N
    np.random.RandomState()
    par = 0.25*np.random.randn(2 * (N + M + N * M))
    # Change to a,b,w
    num = N + M + N * M
    parC = np.vectorize(complex)(par[:num], par[num:])
    a = parC[:N]
    b = parC[N:N + M]
    w = parC[N + M:].reshape(M, N)
    w = np.array(w).T
    rbmOrderedDict = OrderedDict([('a', a), ('b', b), ('w', w)])
    # Save parameters so they can be loaded into the netket machine
    with open("Logs/par" + str(par[0]) + ".json", "wb") as output:
        dump(rbmOrderedDict, output)
    # Load into ma
    ma.load("Logs/par" + str(par[0]) + ".json")
    return par

# One Run
# alpha = 1
# N=5
# B = 1
# A = 1
# M = alpha*N
# #N0 = N/2
# # List of Ak
# Ak = []
# for i in range(N - 1):
#     #Ak_i = A / (N0) * np.exp(-i / N0)
#     Ak_i = 1
#     Ak.append(Ak_i)
# # Define hamiltonian and hilbert space
# ha, hi = CSHam(N,B,Ak)
#
# # # Exact Diagonalization
# e,v = exactDiagonalization(ha)
# # Ground state energy
# edEng = e[0]
# # Ground state
# edState = v[0]
#

#
#
# engTemp, stateTemp, runTimeTemp = runDescentCS(N,B,Ak,alpha)
# print('edState ' , edState)
# print('State ' , stateTemp)
# errSR = err(stateTemp, edState, engTemp, edEng,N)
# print('Eng error ', errSR[0])
# print('State error ', errSR[1])

#
# Parameters
alpha = 1

BValues = np.arange(0, 2, 0.1)
eng = []
for i in range(len(BValues)):
    # Hamiltonian Parameters
    N = 7
    B = BValues[i]
    #B = N/2
    #A = N/2
    #N0 = N / 2
    M = alpha*N
    # List of Ak
    Ak = []
    for i in range(N - 1):
        #Ak_i = A / (N0) * np.exp(-i / N0)
        Ak_i = 1
        Ak.append(Ak_i)
    # Define hamiltonian and hilbert space
    ha, hi = CSHam(N,B,Ak)

    # # Exact Diagonalization
    #start = time.time()
    e,v = exactDiagonalization(ha)
    #end = time.time()
    #runTime = end - start
    # Ground state energy
    edEng = e[0]
    # Ground state
    edState = v[0]
    eng.append(e)


# plt.figure(figsize=(8, 12))
# plt.title("Energy Spectrum Var A\n N=7", size=20)
# plt.plot(BValues, eng)
# plt.ylabel("Eigenstate Energy", size=15)
# plt.xlabel("B", size=15)
# plt.legend(loc = (-0.1, -0.15),fontsize = 12,ncol=4)
# plt.show()

index = 5
print(eng[index])

    # # Lists for Histogram Data
    # numRuns = 1
    # hisIt = np.arange(numRuns)
    # engErr = []
    # stateErr = []
    # runTime = []
    #
    # # Node Information
    # ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=50))
    # pool = mp.Pool(processes=ncpus)
    # # Run Descent
    # resultsSR = [pool.apply(runDescentCS, args=(N,B,Ak,alpha)) for x in hisIt]
    #
    # # Get errors for each run in histogram
    # for i in range(len(hisIt)):
    #     print(resultsSR[i])
    #     engTemp, stateTemp, runTimeTemp = resultsSR[i]
    #     runTime.append(runTimeTemp)
    #     print(edState, np.asmatrix(stateTemp))
    #     errSR = err(np.asmatrix(stateTemp), edState, engTemp, edEng,N)
    #     engErr.append(errSR[0])
    #     stateErr.append(errSR[1])
    # print('Eng error ', engErr)
    # print('State error ', stateErr)
    #
    #
    # #Save data to JSON file
    # data = [engErr, stateErr, runTime]
    # fileName = "Data/21-02-09/N"+str(N)+"M" + str(M)+".json"
    # open(fileName, "w").close()
    # with open(fileName, 'a') as file:
    #     for item in data:
    #         line = json.dumps(item)
    #         file.write(line + '\n')
    # print('SAVED')

    # #Save data to JSON file
    # data = [runTime]
    # fileName = "Data/21-02-09/varexactRunTimeN"+str(N)+"M" + str(M)+".json"
    # open(fileName, "w").close()
    # with open(fileName, 'a') as file:
    #     for item in data:
    #         line = json.dumps(item)
    #         file.write(line + '\n')
    # print('SAVED')
