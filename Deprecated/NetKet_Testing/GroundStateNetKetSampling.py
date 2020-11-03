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
from time import time
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
        ti = time()
        result = f(*args, **kw)
        tf = time()
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

def runDescent(N, M, B, A0):
    par = ranRBMpar(N, M)
    # cgd = CgdConvergance(N, M, B, A0)
    conGradDescent = ConGradDescent(N, B, A0)
    cgd = conGradDescent(N, M, par)
    return cgd

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

def samplingNetKet(n_samples, sampler):
    n_discard = 0.1*n_samples
    batch_size = sampler.sample_shape[0]
    print(batch_size)
    n_samples_chain = int(np.ceil((n_samples / batch_size)))
    n_samples_node = int(np.ceil(n_samples_chain / nk.MPI.size()))
    # Burnout phase
    for _ in sa.samples(n_discard):
        pass
    sam = np.ndarray((n_samples_node, batch_size, ha.hilbert.size))
    # Generate samples and store them
    for i, sample in enumerate(sa.samples(n_samples_node)):
        sam[i] = sample
    return sam

def energyNetKet(sam):
    loc = np.empty(sam.shape[0:2], dtype=np.complex128)
    for i, sample in enumerate(sam):
        _local_values(ha, ma, sample, out=loc[i])

    eloc, loss_stats = loc, _statistics(loc)

    return eloc

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

def energy(par, N, M, H, basis, v):
    v = v.dag()
    psiM = RBM_ansatz(par, N, M, basis)
    E = v*H*psiM
    norm = v.overlap(psiM)
    Enorm = E/norm
    return Enorm.full()[0][0]


# Model Parameters
B=0
A=1
N = 2
M=2
alpha = int(N/M)
ha,hi = hamiltonianNetKet(N, B, A)
# Define machine
ma = nk.machine.RbmSpin(alpha = alpha, hilbert=hi, use_visible_bias = True, use_hidden_bias = True)
#ma.init_random_parameters(sigma=1)
# Define sampler
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=20)
# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)

# Create Basis and Hamiltionian
basis = basisCreation(N)
H = hamiltonian(N, B, A)

# # Random Paramters
# par = [ 0.85854172,  0.8938792,  -0.48961718, -0.0655648,
#  0.70362928,  0.1468188, 0.24020728, -0.96219565,
#  0.21002109, -0.46847805, 0.09819857, 0.30697846,
#   -0.94354464, -0.59374761, -0.55561891, -0.29984356]
#
# parNetKet = [ 0.85854172,  0.8938792,  -0.48961718, -0.0655648,
#  0.70362928, 0.24020728 , 0.1468188, -0.96219565,
#  0.21002109, -0.46847805, 0.09819857, 0.30697846,
#  -0.94354464, -0.55561891, -0.59374761, -0.29984356]

# RBM Parameters for the ground state
# par = [ 6.69376929e-01, 7.28789896e-01, 1.37465157e-03, -8.00709960e-02,
#   2.03412813e+00, -2.03629535e+00, -3.31559288e-01,  4.16797633e-01,
#   5.85959883e-01,  5.36988405e-01, -1.56979953e+00,  7.01433662e-02,
#   1.11482147e+00, -1.11657539e+00, -6.79137466e-01, -6.91849655e-01]
#
# parNetKet = [ 6.69376929e-01, 7.28789896e-01, 1.37465157e-03, -8.00709960e-02,
#   2.03412813e+00, -3.31559288e-01, -2.03629535e+00,  4.16797633e-01,
#   5.85959883e-01,  5.36988405e-01, -1.56979953e+00,  7.01433662e-02,
#   1.11482147e+00, -6.79137466e-01 , -1.11657539e+00, -6.91849655e-01]
# # #
# # # RBM Parameters for the ground state NetKet
# parRBM = [ 6.69376929e-01/2, 7.28789896e-01/2, 1.37465157e-03, -8.00709960e-02,
#   2.03412813e+00/2, -2.03629535e+00/2, -3.31559288e-01/2,  4.16797633e-01/2,
#   5.85959883e-01,  5.36988405e-01, -1.56979953e+00,  7.01433662e-02,
#   1.11482147e+00/2, -1.11657539e+00/2, -6.79137466e-01, -6.91849655e-01]


#RBM Parameters for up up
# par = [ 100, 100, 0, 0,
#   0, 0, 0,  0,
#   0,  0, 0,  0,
#   0, 0, 0, 0]



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
with open("../../Data/07-28-20/paramsGS.json", "wb") as output:
    dump(rbmOrderedDict, output)
# Load into ma
ma.load("Data/07-10-20/paramsGS.json")

it = hi.states().__iter__()
batch_states = np.zeros((512, hi.size))
for i in range(512):
    try:
        batch_states[i] = next(it)
    except StopIteration:
        batch_states.resize(i, hi.size)
        break

print('batch_states ', batch_states )
state1 = ma.log_val(batch_states)
print('log state ', state1)
logmax = np.max(state1.real)
print('logMax ', logmax)
print('log state - logMax ', state1 - logmax)
state = np.exp(state1 - logmax)
#print('state ', state)
#print('state without shift \n ', np.exp(state1))

print('Machine Parameters: ', ma.state_dict())
maArray = ma.to_array()
print('Machine Array: ',maArray)
rbmNetKet = maArray[3]*basis[0][0]+ maArray[2]*basis[0][1]+ maArray[1]*basis[0][2]+ maArray[0]*basis[0][3]
print('NetKet RBM Vector ', rbmNetKet)
rbmVector = RBM_ansatz(par,N,M,basis)
print('RBM Vector: ', rbmVector)
overlap = rbmVector.overlap(rbmNetKet)
print('Over lap of machine vectors ', overlap)



Sbasis = basis[0]
rbm0 = Sbasis[0].dag()*rbmVector
rbm0Norm = (np.abs(rbm0.full()[0][0]))**2
rbm1 = Sbasis[1].dag()*rbmVector
rbm1Norm = (np.abs(rbm1.full()[0][0]))**2
rbm2 = Sbasis[2].dag()*rbmVector
rbm2Norm = (np.abs(rbm2.full()[0][0]))**2
rbm3 = Sbasis[3].dag()*rbmVector
rbm3Norm = (np.abs(rbm3.full()[0][0]))**2
rbmNorm = [rbm0Norm,rbm1Norm,rbm2Norm,rbm3Norm]
exactEnergy = varEnergy(par, N, M, H, basis)


# Create Samples
sam = samplingNetKet(1000, sa)
states = configState(sam, basis)
print('Number of samples ', len(states))
localEngList = []
for i in range(len(states)):
    localEnergy = energy(par, N, M, H, basis, states[i])
    localEngList.append(localEnergy)
mhEnergy = np.sum(localEngList)/len(localEngList)
print('Sampled Energy: ', mhEnergy)
print('Exact Energy: ', exactEnergy)
exactEnergyShort = "%.4f" % exactEnergy
mhEnergyReal = "%.4f" % np.real(mhEnergy)
mhEnergyIm = "%.4f" % np.imag(mhEnergy)
engString = 'Exact Energy: '+ str(exactEnergyShort)+ '    Estimated Energy: ' + str(mhEnergyReal)+'+'+str(mhEnergyIm)+"i"

uu=0
du=0
ud=0
dd=0
#Order uu = 0, ud = 1, du = 2, dd = 3
samLabeled = []
for j in range(len(sam)):
    for k in range(len(sam[0])):
        if sam[j][k][0] == 1:
            if sam[j][k][1] == 1:
                uu += 1
                samLabeled.append(0)
            if sam[j][k][1] == -1:
                ud += 1
                samLabeled.append(1)
        if sam[j][k][0] == -1:
            if sam[j][k][1] == 1:
                du += 1
                samLabeled.append(2)
            if sam[j][k][1] == -1:
                dd += 1
                samLabeled.append(3)

print('uu,ud,du,dd: ', uu,ud,du,dd)
samList = [uu,ud,du,dd]
names = [r'$ \uparrow\uparrow $',r'$\uparrow\downarrow$',r'$\downarrow\uparrow$',r'$\downarrow\downarrow$']
#
plt.figure(figsize=(8,8))
ttl = plt.suptitle("Comparison of NetKet Sampling Results and Expected Distribution",size =15)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.92])

ax2 = plt.subplot(gs[0, :])
ax2.bar([-0.25,0.75, 1.75, 2.75],samList, color = 'red', width=0.5)
ax2.set_xlabel("$\sigma$",size = 12)
ax2.set_ylabel("Number of Samples",size = 12, color='r')
ax2.set_xticks([0,1, 2, 3])
ax2.set_xlim(-0.5,3.5)
ax2.set_xticklabels(names)
ax2.tick_params(axis='y', labelcolor='r')


ax3 = ax2.twinx()
ax3.bar([0.25,1.25, 2.25, 3.25],rbmNorm, color = 'blue', width=0.5)
ax3.set_ylabel("$|\Psi(\sigma)|^2$",size = 12, color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.text(-0.5, -0.09, engString, fontsize=12)
plt.show()

# Many Runs
hisInt=np.arange(10)
ee=[]
mh=[]

for j in range(len(hisInt)):
    exactEnergy = varEnergy(par, N, M, H, basis)
    # Create Samples
    sa.reset()
    sam = samplingNetKet(1000, sa)
    print('sampler[0]', sam[0])
    vector = configState(sam, basis)
    localEngList = []
    for i in range(len(states)):
        localEnergy = energy(par, N, M, H, basis, vector[i])
        localEngList.append(localEnergy)
    mhEnergy = np.sum(localEngList) / len(localEngList)
    mh.append(mhEnergy)
    ee.append(exactEnergy)


labels = ['Exact Energy','Sampled Energy']
plt.figure(constrained_layout=True)
plt.figure(figsize=(8,8))
ttl = plt.suptitle("Comparision of Sampling Energy Estimate and Exact Calculation ",size =15)
gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
ttl.set_position([.5, 0.92])

ax2 = plt.subplot(gs[0, :])
ax2.plot(hisInt, ee, color = 'red', label=labels[0])
ax2.plot(hisInt, np.real(mh), color = 'blue', label=labels[1])
ax2.set_xlabel("Run",size = 12)
ax2.set_ylabel("Energy",size = 12)

ax2.legend(labels, loc = (0.2, -0.1),fontsize = 12,ncol=3)

plt.show()
print('Exact Energy: ', exactEnergy)
print('Energy List ', mh)
#
#
#
#
#














