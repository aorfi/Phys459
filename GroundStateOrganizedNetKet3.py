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
    print('N', N)
    g = nk.graph.Hypercube(length=3, n_dim=1, pbc=False)
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
def err(state, edState, eng, edEng):
    engErr = np.abs(eng - edEng)
    overlap = np.dot(state.conj().T, edState)
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

# *****   Running information
alpha = 1
N=2
B = 1
A = 1
M = alpha*N
# List of Ak
Ak = []
for i in range(N - 1):
    Ak_i = 1
    Ak.append(Ak_i)
ha, hi = CSHam(N,B,Ak)

ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
numRuns = 500
paramAllNK = []
for i in range(numRuns):
    ma.init_random_parameters(seed=None, sigma=0.25)
    param = parameterOutputList(ma)
    paramAllNK.extend(param)
print('Num', len(paramAllNK))



plt.figure(figsize=(10,10))

ttl = plt.suptitle("RBM Parameter Initalization " ,size =20)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.94])
ax1 = plt.subplot(gs[0, 0])
ax1.hist(paramAllNK, bins=np.arange(min(paramAllNK), max(paramAllNK) + 0.08, 0.08), color = 'blue')
x_axis = np.arange(-10, 10, 0.001)
# # Mean = 0, SD = 0.25
ax1.plot(x_axis, 15000*0.08*norm.pdf((x_axis),0,0.25), color = 'red')
ax1.set_xlim(-1.5,1.5)
plt.legend()
plt.show()

# # Ground state energy
# e,v = exactDiagonalization(ha)
# edEng = e[0]
# # Ground state
# edState = v[0]
#
# engTemp, stateTemp, runTimeTemp = runDescentCS(N,B,Ak,alpha)
# errSR = err(stateTemp, edState, engTemp, edEng)
# print('Eng Error: ', errSR[0])
# print('State Error: ', errSR[1])

# # Parameters
# alpha = 1
# # List of N values
# NList = np.arange(3,5)
#
#
# for i in range(len(NList)):
#     # Hamiltonian Parameters
#     N = NList[i]
#     B = 1
#     A = 1
#     M = alpha*N
#     #N0 = N/2
#     # List of Ak
#     Ak = []
#     for i in range(N - 1):
#         #Ak_i = A / (N0) * np.exp(-i / N0)
#         Ak_i = 1
#         Ak.append(Ak_i)
#     # Define hamiltonian and hilbert space
#     ha, hi = CSHam(N,B,Ak)
#
#     # # Exact Diagonalization
#     e,v = exactDiagonalization(ha)
#     # Ground state energy
#     edEng = e[0]
#     # Ground state
#     edState = v[0]
#
#     # Lists for Histogram Data
#     numRuns = 1
#     hisIt = np.arange(numRuns)
#     engErr = []
#     stateErr = []
#     runTime = []
#
#     # Node Information
#     ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=50))
#     pool = mp.Pool(processes=ncpus)
#     # Run Descent
#     resultsSR = [pool.apply(runDescentCS, args=(N,B,Ak,alpha)) for x in hisIt]
#
#     # Get errors for each run in histogram
#     for i in range(len(hisIt)):
#         engTemp, stateTemp, runTimeTemp = resultsSR[i]
#         runTime.append(runTimeTemp)
#         errSR = err(stateTemp, edState, engTemp, edEng)
#         engErr.append(errSR[0])
#         stateErr.append(errSR[1])
#     print('Eng error ', engErr)
#     print('State error ', stateErr)


    # #Save data to JSON file
    # data = [engErr, stateErr, runTime]
    # fileName = "Data/12-01-20/csN"+str(N)+"M" + str(M)+".json"
    # open(fileName, "w").close()
    # with open(fileName, 'a') as file:
    #     for item in data:
    #         line = json.dumps(item)
    #         file.write(line + '\n')
    # print('SAVED')
