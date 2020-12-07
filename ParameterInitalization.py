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

def parameterOutputList(machine):
    aReal = machine.state_dict()['a'].real
    aImag = machine.state_dict()['a'].imag
    bReal = machine.state_dict()['b'].real
    bImag = machine.state_dict()['b'].imag
    wReal = machine.state_dict()['w'].real.reshape([N * alpha * N])
    wImag = machine.state_dict()['w'].imag.reshape([N * alpha * N])
    parameters = np.concatenate((aReal,aImag,bReal,bImag,wReal,wImag)).tolist()
    return parameters

# NetKet RBM with stochastic reconfiguration descent
class RBM:
    def __init__(self, N, hamiltonian, hilbertSpace, machine):
        # Assign inputs
        self.hamiltonian, self.hilbertSpace, self.machine, self.N = hamiltonian, hilbertSpace, machine, N
        # Define sampler
        self.sampler = nk.sampler.MetropolisLocal(machine=self.machine)
        # Define optimizer
        self.optimizer = nk.optimizer.Sgd(learning_rate=0.05)

    # Output is the name of the output file in which the descent data is stored
    def __call__(self, output):
        gs = nk.variational.Vmc(hamiltonian=self.hamiltonian,
                                sampler=self.sampler,
                                optimizer=self.optimizer,
                                # Number of samples use in the MC estimation
                                n_samples=1000,
                                use_iterative=True,
                                method='Sr')
        # Start timing
        start = time.time()
        # Set the output files as well as number of iterations in the descent
        gs.run(output_prefix=output, n_iter=1000)
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
        maParameters = parameterOutputList(self.machine)
        return finalEng, state, runTime,maParameters

# Error Calculation (Input: the found state, the state from exact diagonalization, the found energy, the energy from exact diagonalization)
def err(state, edState, eng, edEng):
    engErr = np.abs(eng - edEng)
    overlap = np.dot(state.conj().T, edState)
    waveFunctionErr = 1 - np.linalg.norm(overlap)
    return engErr, waveFunctionErr


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




# Combines all steps into a function to run on the cluster
def runDescentCS(N,B,Ak,alpha):
    # Define hamiltonian and hibert space (need to do this here cause can't use netket objects as input to use multiprocessing functions)
    ha, hi = CSHam(N,B,Ak)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    # Initialize the RBM parameters (Gaussian with sd 0.25)
    ma.init_random_parameters(0.25)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma)
    # Run RBM
    eng, state, runTime, maParameters = rbm("Logs/CS"+str(N))
    return eng, state, runTime, maParameters



# *****   Running information
N = 4
alpha = 1
M = alpha*N
B = 1
Ak = []
for i in range(N - 1):
    Ak_i = 1
    Ak.append(Ak_i)
ha, hi = CSHam(N,B,Ak)
ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
numRuns = 100
paramAll = []
for i in range(numRuns):
    ranPar(N, alpha, ma)
    param = parameterOutputList(ma)
    paramAll.extend(param)
paramAllNK = []
for i in range(numRuns):
    ma.init_random_parameters(0.25)
    param = parameterOutputList(ma)
    paramAllNK.extend(param)
print(paramAllNK)
# paramAllOutput = []
# for i in range(numRuns):
#     eng, state, runTime, maParameters = runDescentCS(N,B,Ak,alpha)
#     paramAllOutput.extend(maParameters)
#
# data = [paramAll, paramAllNK,paramAllOutput]
# fileName = "Data/12-01-20/ParametersN"+str(N)+"M" + str(M)+".json"
# open(fileName, "w").close()
# with open(fileName, 'a') as file:
#     for item in data:
#         line = json.dumps(item)
#         file.write(line + '\n')
# print('SAVED')

num = len(paramAllNK)
params = [paramAll, paramAllNK]
plt.figure(figsize=(10,10))
labels = ['Constructed Distribution','NetKet Distribution']
colors = ['blue', 'red']
ttl = plt.suptitle("RBM Parameter Initalization " ,size =20)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.94])
ax1 = plt.subplot(gs[0, 0])
ax1.hist(params, bins=15, color = colors, label=labels)
ax1.set_xlim(-2,2)
ax1.legend(labels, loc = (0.2, -0.1),fontsize = 12,ncol=2)
plt.show()

