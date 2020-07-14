from qutip import *
import numpy as np
import scipy
import cmath
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib import gridspec
import itertools

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

# Construct Hamiltonian
def hamiltonian(N, B, A0):
    sx_list = operatorCreation(N)[0]
    sy_list = operatorCreation(N)[1]
    sz_list = operatorCreation(N)[2]
    H = B * sz_list[0]
    for n in range(N - 1):
        H += A0 * sz_list[0] * sz_list[n + 1] + A0 * sx_list[0] * sx_list[n + 1] + A0 * sy_list[0] * sy_list[n + 1]
    return H

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

# Converts config list to vector
def configState(configList,basis):
    N = len(configList)
    spin = np.full(N,0)
    for i in range(N):
        if configList[i] == (-0.5+0j):
            spin[i] = 1
        if configList[i] == (0.5+0j):
            spin[i] = 0
        index = 0
    for i in range(N):
        index += 2**(i)*spin[N-1-i]
    psi = basis[0][index]
    return psi

def proposedState(prevState,N,basis):
    sz_list = operatorCreation(N)[2]
    # Find sz values of previous state
    sz = np.zeros( N, dtype=complex)
    for j in range(N):
        # matrix element <bra|Sz|ket>
        sz[j] = sz_list[j].matrix_element(prevState, prevState)
    # Find position of flip
    flipIndex = int(np.random.uniform(0,N))
    # Flip that spin
    if sz[flipIndex] == (0.5+0j):
        sz[flipIndex] = (-0.5+0j)
    else:
        sz[flipIndex] = (0.5+0j)
    newVec = configState(sz, basis)
    return newVec

def metropolisHastings(prevState,RBMVector, N,basis):
    propState = proposedState(prevState,N,basis)
    propRBM = RBMVector.overlap(propState)
    propSqr = np.conj(propRBM)*propRBM
    prevRBM = RBMVector.overlap(prevState)
    prevSqr = np.conj(prevRBM)*prevRBM
    rbmRatio = np.real(propSqr/prevSqr)
    acceptanceProb = np.min([1,rbmRatio])
    uniform = np.random.uniform()
    if uniform <= acceptanceProb:
        newState = propState
    else:
        newState = prevState
    print(newState)
    return newState

def generateSamples(n_samples, rbmVector, N, basis):
    n_discard = int(0.1 * n_samples)
    initState = basis[0][0]
    discardSamples = [initState]
    for i in range(n_discard):
        newSample = metropolisHastings(discardSamples[i], rbmVector, N, basis)
        discardSamples.append(newSample)
    samples = [discardSamples[-1]]
    for i in range(n_samples-n_discard):
        newSample = metropolisHastings(samples[i], rbmVector, N, basis)
        samples.append(newSample)
    samplesAdd=0
    for i in range(len(samples)):
        samplesAdd += samples[i]
    samplesAddNorm = samplesAdd/len(samples)
    return samplesAddNorm, np.real(samplesAdd.full())

def energySampler(par, N, M, H, basis, v):
    v = v.dag()
    print('v', v)
    psiM = RBM_ansatz(par, N, M, basis)
    print('psiM ', psiM)
    E = v*H*psiM
    vPsiM = v.overlap(psiM)
    Enorm = E/vPsiM
    return Enorm.full()[0][0]


# Model Parameters
B=0
A=1
N = 2
M=2
H = hamiltonian(N,B,A)
basis = basisCreation(N)

# Random RBM Parameters
# par = [ 0.85854172,  0.8938792,  -0.48961718, -0.0655648,   0.70362928,  0.1468188,
#   0.24020728, -0.96219565,  0.21002109, -0.46847805,  0.09819857,  0.30697846,
#  -0.94354464, -0.59374761, -0.55561891, -0.29984356]

par = [ 0, 0, 0, 0,
  0, 0, 0,  0,
  0,  0, 0,  -1*np.pi/4,
  (7/2)*np.pi, -(1/2)*np.pi, (1/2)*np.pi, 0]

rbmVector = RBM_ansatz(par,N,M,basis)
print('RBM Vector: ', rbmVector)

sam = generateSamples(1000, rbmVector, N, basis)
print(np.array(sam[1]))
samList = []
for i in range(len(sam[1])):
    samList.append(sam[1][i][0])
print(samList)

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
print(rbmNorm)


names = [r'$ \uparrow\uparrow $',r'$\uparrow\downarrow$',r'$\downarrow\uparrow$',r'$\downarrow\downarrow$']
#
plt.figure(figsize=(8,8))
ttl = plt.suptitle("Comparison of Sampling Results and Expected Distribution",size =15)
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
plt.show()

# # Many Runs
# hisInt=np.arange(50)
# ee=[]
# mh=[]
# exactEnergy = varEnergy(par, N, M, H, basis)
# for j in range(len(hisInt)):
#     sam = generateSamples(1000, rbmVector, N, basis)
#     mhEnergy = energySampler(par, N, M, H, basis, sam[0])
#     mh.append(mhEnergy)
#     ee.append(exactEnergy)
#
# labels = ['Exact Energy','Sampled Energy']
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(8,8))
# ttl = plt.suptitle("Comparision of Sampling Energy Estimate and Exact Calculation ",size =15)
# gs = gridspec.GridSpec(ncols=1, nrows=1, hspace = 0.4)
# ttl.set_position([.5, 0.92])
#
# ax2 = plt.subplot(gs[0, :])
# ax2.plot(hisInt, ee, color = 'red', label=labels[0])
# ax2.plot(hisInt, np.real(mh), color = 'blue', label=labels[1])
# ax2.set_xlabel("Run",size = 12)
# ax2.set_ylabel("Energy",size = 12)
#
# ax2.legend(labels, loc = (0.2, -0.1),fontsize = 12,ncol=3)
#
# plt.show()
