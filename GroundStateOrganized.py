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
    ma.init_random_parameters(0.25)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma)
    # Run RBM
    eng, state, runTime = rbm("Logs/CS"+str(N))
    return eng, state, runTime

# *****   Running information

# Parameters
alpha = 1
# List of N values
NList = np.arange(2,4)


for i in range(len(NList)):
    # Hamiltonian Parameters
    N = NList[i]
    B = 1
    A = 1
    M = alpha*N
    #N0 = N/2
    # List of Ak
    Ak = []
    for i in range(N - 1):
        #Ak_i = A / (N0) * np.exp(-i / N0)
        Ak_i = 1
        Ak.append(Ak_i)
    print(Ak)
    # Define hamiltonian and hilbert space
    ha, hi = CSHam(N,B,Ak)

    # # Exact Diagonalization
    e,v = exactDiagonalization(ha)
    # Ground state energy
    edEng = e[0]
    # Ground state
    edState = v[0]

    # Lists for Histogram Data
    numRuns = 50
    hisIt = np.arange(numRuns)
    engErr = []
    stateErr = []
    runTime = []

    # Node Information
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=50))
    pool = mp.Pool(processes=ncpus)
    # Run Descent
    resultsSR = [pool.apply(runDescentCS, args=(N,B,Ak,alpha)) for x in hisIt]

    # Get errors for each run in histogram
    for i in range(len(hisIt)):
        engTemp, stateTemp, runTimeTemp = resultsSR[i]
        runTime.append(runTimeTemp)
        errSR = err(stateTemp, edState, engTemp, edEng)
        engErr.append(errSR[0])
        stateErr.append(errSR[1])
    print('Eng error ', engErr)
    print('State error ', stateErr)


    #Save data to JSON file
    data = [engErr, stateErr, runTime]
    fileName = "Data/12-01-20/csN"+str(N)+"M" + str(M)+".json"
    open(fileName, "w").close()
    with open(fileName, 'a') as file:
        for item in data:
            line = json.dumps(item)
            file.write(line + '\n')
    print('SAVED')
