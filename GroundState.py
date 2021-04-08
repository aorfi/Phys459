# Activate the qutip-env-linux environment (source qutip-env-linux/bin/activate)
# If running on the cluster then you also need mpi4py which is preloaded on Beluga (module load mpi4py)
import netket as nk
import json
from qutip import *
import numpy as np
import time
import multiprocessing as mp
from collections import OrderedDict
from pickle import dump
import os
import matplotlib.pyplot as plt
import scipy
from matplotlib import gridspec
from functools import reduce
plt.style.use('seaborn')
from scipy.stats import norm



# Central Spin Hamiltonian and Hilbert space, inputs are Hamiltonian parameters
def CSHam(N, B, Ak):
    # Make graph with of length N with no periodic boundary conditions
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hilbertSpace = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
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
    hamiltonian = nk.operator.LocalOperator(hilbertSpace, operators=operators, acting_on=sites, dtype=complex)
    return hamiltonian, hilbertSpace

def exactDiagonalization(hamiltonian):
    # Changes Hamiltonian to matrix form
    haMatrix = hamiltonian.to_dense()
    # Gets eigenvalues and vectors
    eigenValues, v = np.linalg.eigh(haMatrix)
    # Orders from smallest to largest
    eigenVectors = [v[:, i] for i in range(len(eigenValues))]
    return eigenValues, eigenVectors


# NetKet RBM with stochastic reconfiguration descent
class RBM:
    def __init__(self, N, hamiltonian, hilbertSpace, machine):
        # Assign inputsv[:, i]
        self.hamiltonian, self.hilbertSpace, self.machine, self.N = hamiltonian, hilbertSpace, machine, N
        # Define sampler
        self.sampler = nk.sampler.MetropolisLocal(hilbert=hilbertSpace)
        # Define optimizer
        self.optimizer = nk.optimizer.Sgd(learning_rate=0.05)
        # Define Stochastic reconfiguration
        self.sr = nk.optimizer.SR(diag_shift=0.01)
        # Variational state
        self.vs = nk.variational.MCState(self.sampler, self.machine, n_samples=1000, n_discard=100)
        
    # Output is the name of the output file in which the descent data is stored
    def __call__(self, output):
        self.vs.init_parameters(nk.nn.initializers.normal(stddev=0.25))
        gs = nk.VMC(hamiltonian=self.hamiltonian,optimizer=self.optimizer,variational_state=self.vs,sr = self.sr)
        # Start timing
        start = time.time()
        # Set the output files as well as number of iterations in the descent
        gs.run(out=output, n_iter=1000)
        end = time.time()
        runTime = end - start
        # Import the data from log file
        data = json.load(open(output+'.log'))
        # Extract the relevant information
        #iters = data["Energy"]["iters"]
        energy_RBM = data["Energy"]["Mean"]["real"]
        #finalEng = energy_RBM[-1]
        finalEng = reduce(lambda x,y: x if y is None else y, energy_RBM)
        # Get machine statethe state of the machine as an array
        state = self.vs.to_array()
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
    # RBM Spin Machine
    ma = nk.models.RBM(alpha=1, dtype=complex,use_visible_bias=True, use_hidden_bias=True)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma)
    # Run RBM
    eng, state, runTime = rbm("Logs/CS"+str(N))
    return eng, state, runTime

for i in range(3):
    N = i+2
    B = 1
    # B=N/2
    # A = N/2
    # N0 = N/2
    alpha = 1
    M = alpha*N
    # List of Ak
    Ak = []
    for i in range(N - 1):
        #Ak_i = A / (N0) * np.exp(-i / N0)
        Ak_i = 1
        Ak.append(Ak_i)
    # Define hamiltonian and hilbert space
    ha, hi = CSHam(N,B,Ak)
    #Exact Diagonalization
    e, v = exactDiagonalization(ha)
    
    #Ground state energy
    edEng = e[0]
    # Ground state
    edState = v[0]

    # Lists for Histogram Data
    numRuns = 1
    hisIt = np.arange(numRuns)
    engErr = []
    stateErr = []
    runTime = []

    # Get errors for each run in histogram
    for i in range(len(hisIt)):
        engTemp, stateTemp, runTimeTemp = runDescentCS(N,B,Ak,alpha)
        runTime.append(runTimeTemp)
        errSR = err(np.asmatrix(stateTemp), edState, engTemp, edEng,N)
        engErr.append(errSR[0])
        stateErr.append(errSR[1])
        print('Eng error ', engErr)
        print('State error ', stateErr)

    # #Save data to JSON file
    # data = [engErr, stateErr, runTime]
    # fileName = "Data/21-04-06/N"+str(N)+"M" + str(M)+".json"
    # open(fileName, "w").close()
    # with open(fileName, 'a') as file:
    #     for item in data:
    #         line = json.dumps(item)
    #         file.write(line + '\n')
    # print('SAVED')