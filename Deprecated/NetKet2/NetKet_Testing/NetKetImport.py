import netket as nk
import json
import os
from qutip import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib import gridspec
import itertools
from numpy.random import seed
from scipy import optimize
from functools import wraps
import time

import copy
import pickle
from pickle import load, dump
import collections
from collections import OrderedDict
import multiprocess as mp
from multiprocess.pool import ThreadPool as Pool
import os




# Wrapper to time functions
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ti = time.time()
        result = f(*args, **kw)
        tf = time.time()
        t = tf - ti
        return result, t

    return wrap

# Make basis and get sz values
def operatorCreation(N):
    # operator definitionis
    si = qeye(2)
    sx = 0.5 * sigmax()
    sy = 0.5 * sigmay()
    sz = 0.5 * sigmaz()
    sx_list = []
    sy_list = []
    sz_list = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sy
        sy_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))
        op_list[n] = si
    id = tensor(op_list)
    return sx_list, sy_list, sz_list, id

# Construct Hamiltonian
def hamiltonian(N, B, A0):
    sx_list = operatorCreation(N)[0]
    sy_list = operatorCreation(N)[1]
    sz_list = operatorCreation(N)[2]
    H = B * sz_list[0]
    for n in range(N - 1):
        H += A0 * sz_list[0] * sz_list[n + 1] + A0 * sx_list[0] * sx_list[n + 1] + A0 * sy_list[0] * sy_list[n + 1]
    return H

# Check if dengerate
def CheckDegenerate(H):
    estates = H.eigenstates()
    lowState = estates[0][0]
    secLowState = estates[0][1]
    if (abs(lowState - secLowState)) <= 1e-10:
        return True
    else:
        return False

# Get Ground State Energy and Wavefuntion
class GroundState:

    def __init__(self, N, B, A0):
        self.hamiltonian = hamiltonian(N, B, A0)

    @timing
    def __call__(self):
        # find ground state
        H = self.hamiltonian
        groundState = H.groundstate()
        return groundState[0], groundState[1]

# Make basis and get sz values
def basisCreation(N):
    sz_list = operatorCreation(N)[2]
    Sbasis = []
    basisState = []

    for j in range(2):
        basisState.append(basis(2, j))
    b = itertools.product(basisState, repeat=N)
    basisTensor = list(b)
    # makes Sbasis the correct dimesion of Qobj
    for i in range(2 ** N):
        c = basisTensor[i][0]
        for j in range(N - 1):
            c = tensor(c, basisTensor[i][j + 1])
        Sbasis.append(c)

    # get sz values for basis states
    sz = np.zeros((2 ** N, N), dtype=complex)
    a = [[1 for j in range(N)] for i in range(2 ** N)]
    for i in range(2 ** N):
        for j in range(N):
            # matrix element <bra|Sz|ket>
            sz[i][j] = sz_list[j].matrix_element(Sbasis[i], Sbasis[i])
    return Sbasis, sz

# get randomized RBM parameters (between zero and 1)
def ranRBMpar(N, M):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    par = 1 - 2 * np.random.rand(2 * (N + M + N * M))
    return par

# Function to give RBM wavefuntion
def RBM_ansatz(par, N, M,basis):
    Sbasis = basis[0]
    sz = basis[1]
    # make parmeters complex
    num = N + M + N * M
    parC = np.vectorize(complex)(par[:num], par[num:])
    a = parC[:N]
    b = parC[N:N + M]
    W = parC[N + M:].reshape(M, N)
    expTerm = np.zeros(2 ** N, dtype=complex)
    coshTerm = np.zeros((M, 2 ** N), dtype=complex)
    psiMValues = np.zeros(2 ** N, dtype=complex)
    psiM = 0 * Sbasis[0]

    for i in range(2 ** N):
        for m in range(M):
            coshTerm[m][i] = 2 * np.cosh(np.dot(W[m], sz[i]) + b[m])
    hidProduct = np.prod(coshTerm, axis=0)

    for i in range(2 ** N):
        expTerm[i] = np.exp(np.dot(a, sz[i]))
        psiMValues[i] = expTerm[i] * hidProduct[i]
        psiM += psiMValues[i] * Sbasis[i]
    psiNorm = psiM.unit()
    return psiNorm

# Variation Energy Definition
def varEnergy(par, N, M, H, basis):
    psiM = RBM_ansatz(par, N, M, basis)
    E = expect(H, psiM)
    norm = psiM.norm() ** 2
    Enorm = E / norm
    return Enorm

# Energy Partials
def gradEnergy(par, N, M, H):
    eps = 0.001
    return optimize.approx_fprime(par, varEnergy, eps, N, M, H)

# Conjugate Gradient Descent
class ConGradDescent:

    def __init__(self, N, B, A0,basis):
        H = hamiltonian(N, B, A0)
        isDeg = CheckDegenerate(H)
        if (isDeg == True):
            raise ValueError("Hamiltonian has degenerate ground state")
        else:
            self.hamiltonian = H
        self.basis = basis

    @timing
    def __call__(self, N, M, par):
        H = self.hamiltonian
        basis = self.basis
        min = scipy.optimize.fmin_cg(varEnergy, par, args=(N, M, H,basis), gtol=1e-04, full_output=True, retall=True,
                                     disp=True)
        # Ground State
        found_gs = RBM_ansatz(min[0], N, M,basis)
        found_gs = found_gs.unit()
        # Ground State Energy
        found_gsEnergy = varEnergy(min[0], N, M, H, basis)
        return min, found_gs, found_gsEnergy

# Error Calculation
def err(found_gs, gs, found_gsEnergy, gsEnergy):
    engErr = np.abs(found_gsEnergy - gsEnergy)
    waveFunctionErr = found_gs.dag() * gs
    waveFunctionErr = 1 - waveFunctionErr.norm()
    return engErr, waveFunctionErr


# **** NetKet RBM ****

#Central Spin Hamiltonian and Hilbert space defined in NetKet objects
def hamiltonianNetKet(N, B, A):
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
        operators.append((A * itOp).tolist())
        sites.append([0, (i+1)])

    print('sites: ', sites)
    print('operators: ', operators)
    ha = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    res = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=False)
    print("NetLEt ground state energy = {0:.3f}".format(res.eigenvalues[0]))
    #Returns Hamiltonian and Hilbert space
    return ha, hi

# Sampler
def samplingNetKet(n_samples, sampler, hamiltonian):
    n_discard = 0.1*n_samples
    batch_size = sampler.sample_shape[0]
    n_samples_chain = int(np.ceil((n_samples / batch_size)))
    n_samples_node = int(np.ceil(n_samples_chain / nk.MPI.size()))
    # Burnout phase
    for _ in sampler.samples(n_discard):
        pass
    sam = np.ndarray((n_samples_node, batch_size, hamiltonian.hilbert.size))
    # Generate samples and store them
    for i, sample in enumerate(sampler.samples(n_samples_node)):
        sam[i] = sample
    return sam

# Changes sampled values to states
def configState(input,basis):
    N = len(input[0][0])
    states = []
    for k in range(len(input)):
        for j in range(len(input[0])):
            spin = np.full(N,0)
            for i in range(N):
                if input[k][j][i] == -1:
                    spin[i] = 1
                if input[k][j][i] == 1:
                    spin[i] = 0
            index = 0
            for i in range(N):
                index += 2**(i)*spin[N-1-i]
            psi = basis[0][index]
            states.append(psi)
    return states

# Calculates Local energy of samples
def energyLocalNetKet(par, N, M, H, basis, v):
    v = v.dag()
    psiM = RBM_ansatz(par, N, M, basis)
    E = v*H*psiM
    norm = v.overlap(psiM)
    Enorm = E/norm
    return Enorm.full()[0][0]

# Exact Digonalization NetKet
def exactDigonalization(ha):
    haMatrix = ha.to_dense()
    e, v = np.linalg.eigh(haMatrix)
    inds = np.argsort(e)
    e = e[inds]
    v = v[:, inds]
    return e, v

# Define Netket RBM
class NetKetRBM:
    def __init__(self,N,ha,hi,alpha, ma):
        self.ha,self.hi, self.ma = ha, hi, ma
        # Define sampler
        self.sa = nk.sampler.MetropolisLocal(machine=self.ma)
        # Optimizer
        self.op = nk.optimizer.Sgd(learning_rate=0.05)

    def __call__(self,basis):
        gs = nk.Vmc(
            hamiltonian=self.ha,
            sampler=self.sa,
            optimizer=self.op,
            n_samples=1000,
            n_discard=None,
            sr=None,
        )
        start = time.time()
        gs.run(output_prefix='RBM', n_iter=600)
        end = time.time()
        runTime = end-start
        # import the data from log file
        data = json.load(open("../RBM.log"))
        # Extract the relevant information
        iters = []
        energy_RBM = []

        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            engTemp = iteration["Energy"]["Mean"]
            energy_RBM.append(engTemp)
        finalEng = energy_RBM[-1]

        maArray = self.ma.to_array()
        finalState = maArray[3] * basis[0][0] + maArray[2] * basis[0][1] + maArray[1] * basis[0][2] + maArray[0] * \
                    basis[0][3]
        return finalEng, finalState,  runTime


class NetKetSR:
    def __init__(self, N, ha, hi, alpha, ma):
        self.ha, self.hi, self.ma = ha, hi, ma
        # Define sampler
        self.sa = nk.sampler.MetropolisLocal(machine=self.ma)
        # Optimizer
        self.op = nk.optimizer.Sgd(learning_rate=0.05)

    def __call__(self, basis):
        gs = nk.variational.Vmc(hamiltonian=self.ha,
                                sampler=self.sa,
                                optimizer=self.op,
                                n_samples=1000,
                                use_iterative=True,
                                method='Sr')
        start = time.time()
        gs.run(output_prefix='RBM', n_iter=600)
        end = time.time()
        runTime = end - start
        # import the data from log file
        data = json.load(open("../RBM.log"))
        # Extract the relevant information
        iters = []
        energy_RBM = []

        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            engTemp = iteration["Energy"]["Mean"]
            energy_RBM.append(engTemp)
        finalEng = energy_RBM[-1]

        maArray = self.ma.to_array()
        finalState = maArray[3] * basis[0][0] + maArray[2] * basis[0][1] + maArray[1] * basis[0][2] + maArray[0] * \
                     basis[0][3]
        return finalEng, finalState, runTime

# Created Random RBM Parameters
def ranRBMParams(N, alpha):
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    ranRbm = nk.machine.RbmSpin(alpha=alpha, hilbert=hi)
    ranRbm.init_random_parameters(sigma=1)
    state = copy.deepcopy(ranRbm.state_dict())
    params = ranRbm.parameters
    return state, params

# Change RBM parameters to netKet RBM paramters, and loads machine
def covertParams(N,M,par, ma):
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
    with open("../../../Data/07-28-20/paramsGS.json", "wb") as output:
        dump(rbmOrderedDict, output)
    # Load into ma
    ma.load("Data/07-28-20/paramsGS.json")

# Cluster Functions

def runDescent(N, M, B, A, par,basis):
    conGradDescent = ConGradDescent(N, B, A, basis)
    cgd = conGradDescent(N, M, par)
    return cgd

def runDescentNK(N, ha, hi, alpha, ma,par,basis):
    covertParams(N, M, par, ma)
    rbmNK = NetKetRBM(N, ha, hi, alpha, ma)
    engNKTemp, stateNKTemp, runTimeNKTemp = rbmNK(basis)
    return engNKTemp, stateNKTemp, runTimeNKTemp

def runDescentSR(N, ha, hi, alpha, ma,par,basis):
    covertParams(N, M, par, ma)
    rbmSR = NetKetSR(N, ha, hi, alpha, ma)
    engSRTemp, stateSRTemp, runTimeSRTemp = rbmSR(basis)
    return engSRTemp, stateSRTemp, runTimeSRTemp


#
# # Hamiltionian Parameters
B=1
A=1
N = 4
# RBM Parameters
# ALPHA NEEDS TO  BE AN INTEGER!!!
alpha = 1
M = alpha*N
basis = basisCreation(N)
H = hamiltonian(N,B,A)
print('Hamiltionian: ', H)

# ** NETKET OBJECTS ***
ha,hi = hamiltonianNetKet(N, B, A)
print('NetKet Hamiltionian: ', ha.to_dense())
it = hi.states().__iter__()
batch_states = np.zeros((512, hi.size))
for i in range(512):
    try:
        batch_states[i] = next(it)
    except StopIteration:
        batch_states.resize(i, hi.size)
        break

print('Hilbert states ', batch_states )

# Define machine
ma = nk.machine.RbmSpin(alpha = alpha, hilbert=hi, use_visible_bias = True, use_hidden_bias = True)
# Define sampler
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=20)
# Create RBM Parameters
randomParams = ranRBMpar(N, M)
# Update NetKet machine with randomParams
covertParams(N, M, randomParams, ma)

# Exact Diagonalization
groundState = GroundState(N, B, A)
ed = groundState()
edEng = ed[0][0]
edState = ed[0][1]

print('Machine Parameters: ', ma.state_dict())
maArray = ma.to_array()
rbmNetKet = maArray[3]*basis[0][0]+ maArray[2]*basis[0][1]+ maArray[1]*basis[0][2]+ maArray[0]*basis[0][3]
print('NetKet RBM Vector ', rbmNetKet)
E = expect(H, rbmNetKet)
norm = rbmNetKet.norm() ** 2
Enorm = E / norm
print('NK Energy From State Before Optimization', Enorm )
rbmVector = RBM_ansatz(randomParams,N,M,basis)
print('RBM Vector: ', rbmVector)
E = expect(H, rbmVector)
norm = rbmVector.norm() ** 2
Enorm = E / norm
print('Energy for input vector', Enorm )

# NetKet Run
rbmNK = NetKetRBM(N, ha, hi, alpha, ma)
engNK, stateNK, runTimeNK= rbmNK(basis)
print('Eng, State, Runtime ', engNK, stateNK, runTimeNK)
errNK = err(stateNK,edState,engNK,edEng)

print('eng error: ', errNK[0])
print('state error: ', errNK[1])


# Get iteration information
data = json.load(open("../RBM.log"))
iters = []
energy_RBM = []
for iteration in data["Output"]:
    iters.append(iteration["Iteration"])
    engTemp = iteration["Energy"]["Mean"]
    energy_RBM.append(engTemp)

# Plot Iteration
fig, ax1 = plt.subplots()
plt.title('NetKet Central Spin Iteration N = 3, M = 3, B = 1, A = 1 ', size=20)
ax1.plot(iters, energy_RBM - edEng, color='red', label='Energy (RBM)')
ax1.set_ylabel('Energy Error')
#ax1.set_ylim(0,1.5)
ax1.set_xlabel('Iteration')

plt.show()



#ma.load("RBM.wf")
print('Machine Parameters: ', ma.state_dict())
maArray = ma.to_array()
rbmNetKet = maArray[3]*basis[0][0]+ maArray[2]*basis[0][1]+ maArray[1]*basis[0][2]+ maArray[0]*basis[0][3]
print('NetKet RBM Vector ', rbmNetKet)
E = expect(H, rbmNetKet)
norm = rbmNetKet.norm() ** 2
Enorm = E / norm
print('NK Energy From State ', Enorm)
print('NK Energy ', energy_RBM[-1])
