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
    np.random.seed()
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
def runDescentCS(N,B,A,alpha):
    # Define hamiltonian and hibert space
    ha, hi = CSHam(N,B,A)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    # Initialize the RBM parameters
    #ranPar(N, alpha, ma)
    seed = 1 - 2 * np.random.rand(1)
    ma.init_random_parameters(seed,1)
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
    ranPar(N, alpha, ma) # THIS IS CHANGED FROM THE NetKet INITIALIZATION
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

def parameterOutputList(machine):
    aReal = ma.state_dict()['a'].real
    aImag = ma.state_dict()['a'].imag
    bReal = ma.state_dict()['b'].real
    bImag = ma.state_dict()['b'].imag
    wReal = ma.state_dict()['w'].real.reshape([N * alpha * N])
    wImag = ma.state_dict()['w'].imag.reshape([N * alpha * N])
    parameters = np.concatenate((aReal,aImag,bReal,bImag,wReal,wImag)).tolist()
    return parameters

# *****   Running information
N = 4
alpha = 1
M = alpha*N
B = 1
A = 1
ha, hi = CSHam(N,B,A)
ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
paramAll = []
for i in range(100):
    ranPar(N, alpha, ma)
    param = parameterOutputList(ma)
    paramAll.extend(param)
paramAllNK = []
for i in range(100):
    ma.init_random_parameters(0.25)
    param = parameterOutputList(ma)
    paramAllNK.extend(param)

num = len(paramAllNK)
params = [paramAll, paramAllNK]
plt.figure(figsize=(10,10))
labels = ['Constructed Distribution','NetKet Distribution']
colors = ['blue', 'green']
ttl = plt.suptitle("RBM Parameter Initalization " ,size =20)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.94])
ax1 = plt.subplot(gs[0, 0])
ax1.hist(params, bins=20, color = colors, label=labels)
ax1.set_xlim(-2,2)
ax1.legend(labels, loc = (0.2, -0.1),fontsize = 12,ncol=2)
plt.show()

