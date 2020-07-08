import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy import optimize
from functools import wraps
import time
import itertools
from scipy.optimize import minimize
from matplotlib import gridspec
import json
import multiprocessing as mp
import os
import scipy
import cmath
import copy
plt.style.use('seaborn')


# **** RBM without NetKet ****

# Wrapper to time functions
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        t = end - start
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
def hamiltonian(N,B,A0):
    sx_list = operatorCreation(N)[0]
    sy_list = operatorCreation(N)[1]
    sz_list = operatorCreation(N)[2]
    H = B*sz_list[0]
    for n in range(N-1):
        H += A0*sz_list[0]*sz_list[n+1] + A0*sx_list[0]*sx_list[n+1] + A0*sy_list[0]*sy_list[n+1]
    return H

# Check if dengerate
def CheckDegenerate(H):
    estates = H.eigenstates()
    lowState = estates[0][0]
    secLowState = estates[0][1]
    if (abs(lowState-secLowState))<= 1e-10:
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
        print(groundState)
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

# Get randomized RBM parameters (between zero and 1)
def ranRBMpar(N,M):
    par = 1-2*np.random.rand(2*(N+M+N*M))
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

#Variation Energy Definition
def varEnergy(par,N, M, H):
    psiM = RBM_ansatz(par,N, M)
    E = expect(H,psiM)
    norm = psiM.norm()**2
    Enorm = E/norm
    return Enorm

#Energy Partials
def gradEnergy(par,N, M,H):
    eps = 0.001
    return optimize.approx_fprime(par,varEnergy,eps, N, M,H)

# Conjugate Gradient Descent
class ConGradDescent:

    def __init__(self, N, B, A0):
        H = hamiltonian(N, B, A0)
        isDeg = CheckDegenerate(H)
        if (isDeg == True):
            raise ValueError("Hamiltonian has degenerate ground state")
        else:
            self.hamiltonian = H

    @timing
    def __call__(self, N, M, par):
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



# **** NetKet RBM ****



# Central Spin Hamiltionian
def hamiltonianCS(N, B, A):
    # Make graph of length N
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
    ha = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    #Returns Hamiltonian and Hilbert space
    return ha, hi

# Heisenberg Hamiltiona
def hamiltonianHeis(N, J):
    # Make graph with no edges of length N
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    sz_sz = np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                       [0, 0, -1, 0], [0, 0, 0, 1]])
    exchange = np.array(
        [[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    heis_term = 0.25*(sz_sz + exchange)
    ha = nk.operator.GraphOperator(hi, bondops=[J * heis_term])
    #Returns Hamiltonian and Hilbert space
    return ha, hi

# Exact Digonalization
def exactDigonalization(ha):
    haMatrix = ha.to_dense()
    e, v = np.linalg.eigh(haMatrix)
    inds = np.argsort(e)
    e = e[inds]
    v = v[:, inds]
    return e, v

# Define Netket RBM
class NetKetRBM:
    def __init__(self,N,ha,hi,alpha):
        self.ha,self.hi = ha, hi
        # Define machine
        self.ma = nk.machine.RbmSpin(alpha = alpha, hilbert= self.hi)
        # Define sampler
        self.sa = nk.sampler.MetropolisLocal(machine=self.ma)
        # Optimizer
        self.op = nk.optimizer.Sgd(learning_rate=0.05)
        e,v = exactDigonalization(self.ha)
        self.exact_gs_energy = e[0]
        print('Ground State: ', self.exact_gs_energy)

    def __call__(self, rbmState):
        # Initialize parameters
        self.ma.load_state_dict(rbmState)
        #self.ma.init_random_parameters(sigma=0.05)
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



# **** Run All RBMs ****



# Hamiltionian Parameters
N= 3
B = 1
A = 1
J = 1

# RBM Parameters
M=3
alpha = int(N/M)

# # Histogram All
# hisIt = np.arange(50)
# # engErrHei = []
# # runTimeHei = []
# engErrCS = []
# runTimeCS = []
# runTime = []
# engErr = []
# # haHei, hiHei = hamiltonianHeis(N, J)
# # print('Heisenberg Hamiltionian: ', haHei.to_dense())
# haCS, hiCS = hamiltonianCS(N, B, A)
# print('Central Spin Hamiltionian: ', haCS.to_dense())
# for i in range(len(hisIt)):
#     # Create RBM Parameters
#     rbmState, rbmParams = ranRBMParams(N, alpha)
#     # Change rbmParams to form that can be used with old RBM
#     rbmParamsConverted = np.concatenate([np.real(rbmParams), np.imag(rbmParams)])
#     # Heisenberg Run
#     # rbmHei = NetKetRBM(N, haHei, hiHei, alpha)
#     # runTimeHeiTemp, engErrHeiTemp = rbmHei(rbmState)
#     # runTimeHei.append(runTimeHeiTemp)
#     # engErrHei.append(engErrHeiTemp)
#     # Central Spin Run
#     rbmCS = NetKetRBM(N, haCS, hiCS, alpha)
#     runTimeCSTemp, engErrCSTemp = rbmCS(rbmState)
#     runTimeCS.append(runTimeCSTemp)
#     engErrCS.append(engErrCSTemp)
#     #Non netket RBM
#     conGradDescent = ConGradDescent(N, B, A)
#     cgd = conGradDescent(N, M, rbmParamsConverted)
#     groundState = GroundState(N, B, A)
#     ed = groundState()m
#     cgdEngTemp = cgd[0][2]
#     edEngTemp = ed[0][0]
#     cgdStateTemp = cgd[0][1]
#     edStateTemp = ed[0][1]
#     cgdErrTemp = err(cgdStateTemp, edStateTemp, cgdEngTemp, edEngTemp)
#     engErr.append(cgdErrTemp[0])
#     runTime.append(cgd[1])
#
# # Save data to JSON file
# data = [engErrCS, engErr, runTimeCS, runTime]
# open("Data/06-18-20/CSComparisonN2B1.json", "w").close()
# with open('Data/06-18-20/CSComparisonN2B1.json', 'a') as file:
#     for item in data:
#         line = json.dumps(item)
#         file.write(line + '\n')
#
# # Plotting
# allEngErr = [engErrCS, engErr]
# allRunTime = [ runTimeCS, runTime]
# labels = ['NetKet Central Spin', 'Non-NetKet RBM']
# colors = ['blue', 'green']
#
# hisIt= np.arange(len(engErrCS))
# plt.figure(constrained_layout=True)
# plt.figure(figsize=(10,10))
# ttl = plt.suptitle("Predefined Heisenberg Hamiltonian N = 2",size =20)
# gs = gridspec.GridSpec(ncols=2, nrows=3, hspace = 0.4)
# ttl.set_position([.5, 0.92])
#
# ax2 = plt.subplot(gs[0, 0])
# ax2.hist(allEngErr, bins=10, color = colors, label=labels)
# ax2.set_xlabel("$\Delta E = |E_{RBM}-E_{ED}|$",size = 15)
#
# ax3 = plt.subplot(gs[0, 1])
# ax3.hist(allRunTime, bins=10, color = colors)
# ax3.set_xlabel("Runtime (s)",size = 15)
#
# ax4 = plt.subplot(gs[1, :])
# #ax4.scatter(hisIt,engErrHei, color = 'red')
# ax4.scatter(hisIt,engErrCS, color = 'blue')
# ax4.scatter(hisIt,engErr, color = 'green')
# #ax4.set_ylim([-0.000005,0.000005])
# ax4 .set_ylabel("$\Delta E = |E_{RBM}-E_{ED}|$", size = 15)
#
# ax2.legend(labels, loc = (0, -3.3),fontsize = 12,ncol=3)
#
# ax5 = plt.subplot(gs[2, :])
# #ax5.scatter(hisIt,runTimeHei, color = 'red')
# ax5.scatter(hisIt,runTimeCS, color = 'blue')
# ax5.scatter(hisIt,runTime, color = 'green')
# ax5.set_xlabel("Run Number",size = 15)
# ax5 .set_ylabel("Runtime (s)", size = 15)
# plt.show()

# One Run
haCS, hiCS = hamiltonianCS(N, B, A)
e,v = exactDigonalization(haCS)
exact_gs_energy = e[0]
iters_All = []
energy_RBM_All = []
for i in range(5):
    # Create RBM Parameters
    rbmState, rbmParams = ranRBMParams(N, alpha)
    rbmCS = NetKetRBM(N, haCS, hiCS, alpha)
    runTimeCSTemp, engErrCSTemp = rbmCS(rbmState)
    # Get iteration information
    data = json.load(open("RBM.log"))
    iters = []
    iters_All.append(iters)
    energy_RBM = []
    energy_RBM_All.append(energy_RBM)
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        engTemp = iteration["Energy"]["Mean"]
        energy_RBM.append(engTemp)

# Plot Iteration
fig, ax1 = plt.subplots()
plt.title('NetKet Central Spin Iteration N = 3, M = 3, B = 1, A = 1 ', size=20)
ax1.plot(iters_All[0], energy_RBM_All[0] - exact_gs_energy, color='red', label='Energy (RBM)')
ax1.plot(iters_All[1], energy_RBM_All[1] - exact_gs_energy, color='blue', label='Energy (RBM)')
ax1.plot(iters_All[2], energy_RBM_All[2] - exact_gs_energy, color='green', label='Energy (RBM)')
ax1.plot(iters_All[3], energy_RBM_All[3] - exact_gs_energy, color='black', label='Energy (RBM)')
ax1.plot(iters_All[4], energy_RBM_All[4] - exact_gs_energy, color='orange', label='Energy (RBM)')
ax1.set_ylabel('Energy Error')
#ax1.set_ylim(0,1.5)
ax1.set_xlabel('Iteration')
#plt.axis([0,iters[-1],exact_gs_energy-0.03,exact_gs_energy+0.2])
plt.show()