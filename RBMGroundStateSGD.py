#!/usr/bin/env python
# coding: utf-8

from qutip import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import itertools
from scipy import optimize
from functools import wraps
from time import time
from netket.optimizer import Sgd

plt.style.use('seaborn')


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
    # operator definitions
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
    iden = tensor(op_list)

    return sx_list, sy_list, sz_list, iden


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
    par = np.random.rand(2 * (N + M + N * M))
    return par


# Function to give RBM wavefuntion
def RBM_ansatz(par, N, M):
    Sbasis = basisCreation(N)[0]
    sz = basisCreation(N)[1]
    # make parmeters complex
    num = N + M + N * M
    parC = np.vectorize(complex)(par[:num], par[num:])
    a = parC[:N]
    b = parC[N:N + M]
    W = parC[N + M:].reshape(M, N)
    expTerm = np.zeros(2 ** N, dtype=complex)
    coshTerm = np.zeros((M, 2 ** N), dtype=complex)
    hidProduct = np.zeros(2 ** N, dtype=complex)
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
def varEnergy(par, N, M, H):
    psiM = RBM_ansatz(par, N, M)
    E = expect(H, psiM)
    norm = psiM.norm() ** 2
    Enorm = E / norm
    return Enorm


# Energy Partials
def gradEnergy(par, N, M, H):
    eps = 0.001
    return optimize.approx_fprime(par, varEnergy, eps, N, M, H)


# Gradient Descent
class GradDescent:

    def __init__(self, N, B, A0):
        H = hamiltonian(N, B, A0)
        isDeg = CheckDegenerate(H)
        if (isDeg == True):
            raise ValueError("Hamiltonian has degenerate ground state")
        else:
            self.hamiltonian = H

    @timing
    def __call__(self, N, M):
        par = ranRBMpar(N, M)
        H = self.hamiltonian
        min = scipy.optimize.fmin_cg(varEnergy, par, args=(N, M, H), gtol=1e-04, full_output=True, retall=True,
                                     disp=True)
        # Ground State
        found_gs = RBM_ansatz(min[0], N, M)
        found_gs = found_gs.unit()
        # Ground State Energy
        found_gsEnergy = varEnergy(min[0], N, M, H)

        return min, found_gs, found_gsEnergy


# Error Calculation
def err(found_gs, gs, found_gsEnergy, gsEnergy):
    engErr = np.abs(found_gsEnergy - gsEnergy)
    waveFunctionErr = found_gs.dag() * gs
    waveFunctionErr = 1 - waveFunctionErr.norm()
    return engErr, waveFunctionErr


# # Test

# Parameter definition
N = 2
M = 2
B = 1
A0 = 1

# gradient descent
gradDescent = GradDescent(N, B, A0)
gd = gradDescent(N, M)
gdResults = gd
gdState = gd[0]
gdTime = gd[1]

# exact diagonalization
groundState = GroundState(N, B, A0)
ed = groundState()
edState = ed[0]
edTime = ed[1]

# Error
gdEng = gd[0][2]
edEng = ed[0][0]
ActualEng = edEng
gdGroundState = gd[0][1]
edGroundState = ed[0][1]
errTemp = err(gdGroundState, edGroundState, gdEng, edEng)
engErr = errTemp[0]
stateErr = errTemp[1]

# Readout of Results
print('Gradient Descent Time: ', gdTime)
print('Exact Diagonalization Time: ', edTime)
print('Energy Energy: ', engErr)
print('Wavefunction Energy: ', stateErr)

# Individual Error vs Iteration
parResults = gdResults[0][0][5]
numIter = len(parResults)
iter = np.arange(numIter)
state = []
stateErr = []

gs = edState[1]
gsEng = ActualEng

# find state at each iteration
for i in range(numIter):
    stateIter = RBM_ansatz(parResults[i], N, M)
    state.append(stateIter)
    errIter = err(stateIter, gs, 0, 0)
    stateErr.append(errIter[1])

plt.plot(iter, stateErr)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('N=' + str(N) + ', M=' + str(M))
plt.show()

op = Sgd(learning_rate=0.05)

