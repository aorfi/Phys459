import multiprocessing as mp
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
import netket as nk
from netket.operator import local_values as _local_values
from netket._core import deprecated
from netket.stats import (
    statistics as _statistics,
    mean as _mean,
)
import copy
import pickle
from pickle import load, dump
import collections
from collections import OrderedDict


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

    def __init__(self, N, B, A0):
        H = hamiltonian(N, B, A0)
        isDeg = CheckDegenerate(H)
        if (isDeg == True):
            raise ValueError("Hamiltonian has degenerate ground state")
        else:
            self.hamiltonian = H
        self.basis = basisCreation(N)

    @timing
    def __call__(self, N, M, par):
        H = self.hamiltonian
        basis = self.basis
        min = scipy.optimize.fmin_cg(varEnergy, par, args=(N, M, H,basis), gtol=1e-04, full_output=True, retall=True,
                                     disp=True)
        # Ground State
        print(min[0])
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
        sites.append([0, (i + 1)])
    print('operators = ', operators)
    print('sites = ', sites)
    ha = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    #Returns Hamiltonian and Hilbert space
    return ha, hi

# Sampler
def samplingNetKet(n_samples, sampler, hamiltonian):
    n_discard = 0.1*n_samples
    batch_size = sampler.sample_shape[0]
    print(batch_size)
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
        e,v = exactDigonalization(self.ha)
        self.exact_gs_energy = e[0]
        print('Ground State: ', self.exact_gs_energy)

    def __call__(self):
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
        data = json.load(open("RBM.log"))
        # Extract the relevant information
        iters = []
        energy_RBM = []

        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            engTemp = iteration["Energy"]["Mean"]
            energy_RBM.append(engTemp)
        finalEng = energy_RBM[-1]
        print("final eng NetKet: ", finalEng)
        engErr = finalEng - self.exact_gs_energy
        return runTime, engErr

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
    print('Saved Paramters: ', rbmOrderedDict)
    # Save parameters so they can be loaded into the netket machine
    with open("Data/07-10-20/paramsGS.json", "wb") as output:
        dump(rbmOrderedDict, output)
    # Load into ma
    ma.load("Data/07-10-20/paramsGS.json")



# Hamiltionian Parameters
B=1
A=1
N = 4
# RBM Parameters
M=4
alpha = int(N/M)

# ** NETKET OBJECTS ***
ha,hi = hamiltonianNetKet(N, B, A)
# Define machine
ma = nk.machine.RbmSpin(alpha = alpha, hilbert=hi, use_visible_bias = True, use_hidden_bias = True)
# Define sampler
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=20)
# Exact Energy
e,v = exactDigonalization(ha)
exact_gs_energy = e[0]

# Histogram All
hisIt = np.arange(50)
engErrNK = []
runTimeNK = []
runTime = []
engErr = []

for i in range(len(hisIt)):
    # Create RBM Parameters
    randomParams = ranRBMpar(N, M)
    # Update NetKet machine with randomParams
    covertParams(N, M, randomParams, ma)

    # Central Spin Run
    rbmNK = NetKetRBM(N, ha, hi, alpha, ma)
    runTimeCSTemp, engErrCSTemp = rbmNK()
    runTimeNK.append(runTimeCSTemp)
    engErrNK.append(engErrCSTemp)

    #Non netket RBM
    conGradDescent = ConGradDescent(N, B, A)
    cgd = conGradDescent(N, M, randomParams)
    groundState = GroundState(N, B, A)
    ed = groundState()
    cgdEngTemp = cgd[0][2]
    edEngTemp = ed[0][0]
    cgdStateTemp = cgd[0][1]
    edStateTemp = ed[0][1]
    cgdErrTemp = err(cgdStateTemp, edStateTemp, cgdEngTemp, edEngTemp)
    engErr.append(cgdErrTemp[0])
    runTime.append(cgd[1])

# Save data to JSON file
data = [engErrNK, engErr, runTimeNK, runTime]
open("Data/07-17-20/N4M4B1.json", "w").close()
with open('Data/07-17-20/N4M4B1.json', 'a') as file:
    for item in data:
        line = json.dumps(item)
        file.write(line + '\n')

# Plotting
allEngErr = [engErrNK, engErr]
allRunTime = [ runTimeNK, runTime]
labels = ['NetKet Central Spin', 'Non-NetKet RBM']
colors = ['blue', 'red']

hisIt= np.arange(len(engErrNK))
plt.figure(constrained_layout=True)
plt.figure(figsize=(10,10))
ttl = plt.suptitle("Comparison of NetKet and Non-NetKet RBM \n N = 4, B = 1, M = 4 ",size =20)
gs = gridspec.GridSpec(ncols=2, nrows=3, hspace = 0.4)
ttl.set_position([.5, 0.94])

ax2 = plt.subplot(gs[0, 0])
ax2.hist(allEngErr, bins=10, color = colors, label=labels)
ax2.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)

ax3 = plt.subplot(gs[0, 1])
ax3.hist(allRunTime, bins=10, color = colors)
ax3.set_xlabel("Runtime (s)",size = 15)

ax4 = plt.subplot(gs[1, :])
ax4.scatter(hisIt,engErrNK, color = 'blue')
ax4.scatter(hisIt,engErr, color = 'red', marker = '^')
ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)

ax2.legend(labels, loc = (0.25, -3.3),fontsize = 12,ncol=3)

ax5 = plt.subplot(gs[2, :])
ax5.scatter(hisIt,runTimeNK, color = 'blue')
ax5.scatter(hisIt,runTime, color = 'red', marker = '^')
ax5.set_xlabel("Run Number",size = 15)
ax5 .set_ylabel("Runtime (s)", size = 15)
plt.show()


## PLOT ONE RUN
# # Get iteration information
# data = json.load(open("RBM.log"))
# iters = []
# energy_RBM = []
# for iteration in data["Output"]:
#     iters.append(iteration["Iteration"])
#     engTemp = iteration["Energy"]["Mean"]
#     energy_RBM.append(engTemp)
#
# # Plot Iteration
# fig, ax1 = plt.subplots()
# plt.title('NetKet Central Spin Iteration N = 3, M = 3, B = 1, A = 1 ', size=20)
# ax1.plot(iters, energy_RBM - exact_gs_energy, color='red', label='Energy (RBM)')
# ax1.set_ylabel('Energy Error')
# #ax1.set_ylim(0,1.5)
# ax1.set_xlabel('Iteration')
# #plt.axis([0,iters[-1],exact_gs_energy-0.03,exact_gs_energy+0.2])
# plt.show()