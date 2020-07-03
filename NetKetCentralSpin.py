import netket as nk
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import time
import json
import multiprocessing as mp
import os


def hamiltonian(N, B, A):
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


class ExactDigonalization:
    def __init__(self,N,B,A):
        self.ha,self.hi = hamiltonian(N,B,A)
        self.N = N

    def __call__(self):
        haMatrix = self.ha.to_dense()
        e, v = np.linalg.eigh(haMatrix)
        inds = np.argsort(e)
        e = e[inds]
        v = v[:, inds]
        return e,v


class RBM:
    def __init__(self,N,B,A,alpha):
        self.ha,self.hi = hamiltonian(N,B,A)
        # Define machine
        self.ma = nk.machine.RbmSpin(alpha = alpha, hilbert= self.hi)
        # Define sampler
        self.sa = nk.sampler.MetropolisLocal(machine=self.ma)
        # Optimizer
        self.op = nk.optimizer.Sgd(learning_rate=0.05)

    def __call__(self):
        # Initialize parameters
        self.ma.init_random_parameters(sigma=1)

        gs = nk.Vmc(
            hamiltonian=self.ha,
            sampler=self.sa,
            optimizer=self.op,
            n_samples=1000,
            n_discard=None,
            sr=None,
        )
        print(gs.info())

        start = time.time()
        gs.run(out='RBM', n_iter=600)
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
        engErr = finalEng - exact_gs_energy
        return runTime, engErr

# Model Parameters
B=1
A=1
N = 2

MList = np.arange(1,2)


ha,hi = hamiltonian(N, B, A)

exact = ExactDigonalization(N,B,A)
evalues, evectors = exact()
exact_gs_energy = evalues[0]
print("ground state energy = ", exact_gs_energy)
print('eigen values = ', evalues)
print("Hamiltonian= \n", ha.to_dense())

# Initialize RBM



# Histogram
hisIt = np.arange(1)

for j in range(len(MList)):
    alpha = int(N / MList[j])
    rbm = RBM(N, B, A, alpha)
    engErr = []
    runTime = []
    for i in range(len(hisIt)):
        runtimeTemp, engErrTemp = rbm()
        runTime.append(runtimeTemp)
        engErr.append(engErrTemp)
    # Save data to JSON file
    # data = [engErr,runTime]
    # fileName = 'Data/06-30-20/sr'+str(N)+'M'+str(MList[j])+'.json'
    # open(fileName, "w").close()
    # with open(fileName, 'a') as file:
    #     for item in data:
    #         line = json.dumps(item)
    #         file.write(line + '\n')



# # One Run
# rbm = RBM(N, B, A, alpha)
# iters_All = []
# energy_RBM_All = []
# for i in range(5):
#     runTime, engErr = rbm()
#     # Get iteration information
#     data = json.load(open("RBM.log"))
#     iters = []
#     iters_All.append(iters)
#     energy_RBM = []
#     energy_RBM_All.append(energy_RBM)
#     for iteration in data["Output"]:
#         iters.append(iteration["Iteration"])
#         engTemp = iteration["Energy"]["Mean"]
#         energy_RBM.append(engTemp)
#
# # Plot Iteration
# fig, ax1 = plt.subplots()
# plt.title('Central Spin N = 2 ', size=20)
# ax1.plot(iters_All[0], energy_RBM_All[0] - exact_gs_energy, color='red', label='Energy (RBM)')
# ax1.plot(iters_All[1], energy_RBM_All[1] - exact_gs_energy, color='blue', label='Energy (RBM)')
# ax1.plot(iters_All[2], energy_RBM_All[2] - exact_gs_energy, color='green', label='Energy (RBM)')
# ax1.plot(iters_All[3], energy_RBM_All[3] - exact_gs_energy, color='black', label='Energy (RBM)')
# ax1.plot(iters_All[4], energy_RBM_All[4] - exact_gs_energy, color='orange', label='Energy (RBM)')
# ax1.set_ylabel('Energy Error')
# #ax1.set_ylim(0,1.5)
# ax1.set_xlabel('Iteration')
# #plt.axis([0,iters[-1],exact_gs_energy-0.03,exact_gs_energy+0.2])
# plt.show()
#
