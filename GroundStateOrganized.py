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


#Central Spin Hamiltonian and Hilbert space defined in NetKet objects
def CSHam(N, B, A):
    # Make graph with of length N with no periodic boundary conditions
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

#Central Spin Hamiltonian and Hilbert with variable Ak (Combine with above)
def CSVarAHam(N, B, A, N0):
    # Make graph with of length N with no periodic boundary conditions
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

# Heisenberg Hamtilonian with Field
def heiHam(N, J, h):
    # Make graph with of length N with no periodic boundary conditions
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

# Exact Digonalization (inputs Netket hamiltonian)
def exactDigonalization(ha):
    # Changes hamiltion to matrix form
    haMatrix = ha.to_dense()
    # Gets eigenvalues and vectors
    e, v = np.linalg.eigh(haMatrix)
    # Orders from smallest to largest
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
        # Define optimizer
        self.op = nk.optimizer.Sgd(learning_rate=0.05)
        self.N = N

    def __call__(self, output):
        gs = nk.variational.Vmc(hamiltonian=self.ha,
                                sampler=self.sa,
                                optimizer=self.op,
                                n_samples=1000,
                                use_iterative=True,
                                method='Sr')
        # Timing
        start = time.time()
        gs.run(output_prefix=output + 'SR', n_iter=1000)
        end = time.time()
        runTime = end - start
        # Import the data from log file
        data = json.load(open(output + "SR.log"))
        # Extract the relevant information
        iters = []
        energy_RBM = []
        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            engTemp = iteration["Energy"]["Mean"]
            energy_RBM.append(engTemp)
        finalEng = energy_RBM[-1]
        # Get machine statethe state of the machine as an array
        state = self.ma.to_array()
        # Outputs the final energy, the final state, and the runtime
        return finalEng, state, runTime

# Error Calculation (Input: the found state, the state from exact diagonalization, the found energy, the energy from exact diagonalization)
def err(state, edState, eng, edEng):
    engErr = np.abs(eng - edEng)
    overlap = np.dot(state.conj().T, edState)
    waveFunctionErr = 1 - np.linalg.norm(overlap)
    return engErr, waveFunctionErr

# Initializes random RBM parameters
def ranPar(N, alpha, ma):
    M = alpha * N
    #Makes list of random parameters between -1,1
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    par = 1 - 2 * np.random.rand(2 * (N + M + N * M))
    # Change to a,b,w
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
    with open("Logs/par" + str(par[0]) + ".json", "wb") as output:
        dump(rbmOrderedDict, output)
    # Load into ma
    ma.load("Logs/par" + str(par[0]) + ".json")
    return par

# Combines all steps into a function to run on the cluster
def runDescentCS(N,B,A,alpha):
    # Define hamiltonian and hibert space
    ha, hi = CSHam(N,B,A)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    # Initialize the RBM parameters
    ranPar(N, alpha, ma)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma)
    # Run RBM
    eng, state, runTime = rbm("Logs/CS"+str(N))
    return eng, state, runTime

def runDescentCSVarA(N, B, A, N0,alpha):
    # Define hamiltonian and hibert space
    ha, hi = CSVarAHam(N,B,A,N0)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    # Initialize the RBM parameters
    ranPar(N, M, ma) # THIS IS CHANGED FROM THE NetKet INITIALIZATION
    # ma.init_random_parameters(1) # NetKet Initialization
    rbm = RBM(N, ha, hi, ma)
    eng, state, runTime = rbm("Logs/CSVarA"+str(N))
    return eng, state, runTime

def runDescentHei(N,J,h,alpha):
    # Define hamiltonian and hibert space
    ha, hi = heiHam(N,J,h)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    # Initialize the RBM parameters
    ranPar(N, alpha, ma)
    rbm = RBM(N, ha, hi, ma)
    eng, state, runTime = rbm("Logs/Hei"+str(N))
    return eng, state, runTime

# *****   Running information

# Parameters
alpha = 1
J=1
h=0.5
# List of N values
NList = np.arange(2,11)

for i in range(len(NList)):
    # Hamiltionian Parameters
    N = NList[i]
    B = 1
    A = 1
    M = alpha*N
    # Define hamiltonian and hilbert space
    ha, hi = CSHam(N,B,A)


    # # Exact Diagonalization
    e,v = exactDigonalization(ha)
    edEng = e[0]
    edState = v[0]

    # Lists for Histogram Data
    numRuns = 1
    hisIt = np.arange(numRuns)
    engErr = []
    stateErr = []
    runTime = []

    # Node Information
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=50))
    pool = mp.Pool(processes=ncpus)
    # Run Descent
    resultsSR = [pool.apply(runDescentCS, args=(N,B,A,alpha)) for x in hisIt]

    # Get errors for each run in histogram
    for i in range(len(hisIt)):
        engTemp, stateTemp, runTimeTemp = resultsSR[i]
        runTime.append(runTimeTemp)
        errSR = err(stateTemp, edState, engTemp, edEng)
        engErr.append(errSR[0])
        stateErr.append(errSR[1])


    # #Save data to JSON file
    # data = [engErr, stateErr, runTime]
    # fileName = "Data/11-17-20/heiN"+str(N)+"M" + str(M)+".json"
    # open(fileName, "w").close()
    # with open(fileName, 'a') as file:
    #     for item in data:
    #         line = json.dumps(item)
    #         file.write(line + '\n')
    # print('SAVED')
