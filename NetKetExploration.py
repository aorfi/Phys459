import sys
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import time
import json
from netket.operator import local_values as _local_values
from netket.stats import (
    statistics as _statistics,
    mean as _mean,
)
from netket._vmc import _estimate_stats
from netket.vmc_common import tree_map



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
B=0
A=1
N = 2
M=2
alpha = int(N/M)

ha,hi = hamiltonian(N, B, A)

exact = ExactDigonalization(N,B,A)
evalues, evectors = exact()
exact_gs_energy = evalues[0]
print("ground state energy = ", exact_gs_energy)
print('eigen values = ', evalues)
print("Hamiltonian= \n", ha.to_dense())


# One Run
rbm = RBM(N, B, A, alpha)


# Define machine
ma = nk.machine.RbmSpin(alpha = alpha, hilbert=hi)
# Define sampler
sa = nk.sampler.MetropolisLocal(machine=ma)
# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)

n_discard = 100
print('n_discard: ', n_discard)
n_samples = 1000
print('n_samples: ', n_samples)
batch_size = sa.sample_shape[0]
print('batch_size: ', batch_size)
n_samples_chain = int(np.ceil((n_samples / batch_size)))
print('n_samples_chain: ', n_samples_chain)
# MPI something with over multiple cores
n_samples_node = int(np.ceil(n_samples_chain / nk.MPI.size()))
print('n_samples_node : ', n_samples_node )

# Burnout phase
for _ in sa.samples(n_discard):
    pass
sam = np.ndarray((n_samples_node, batch_size, ha.hilbert.size))
# Generate samples and store them
for i, sample in enumerate(sa.samples(n_samples_node)):
    sam[i] = sample
print("Samples: ", sam.shape)


loc = np.empty(sam.shape[0:2], dtype=np.complex128)
for i, sample in enumerate(sam):
    _local_values(ha, ma, sample, out=loc[i])
eloc = loc
loss_stats = _statistics(loc)

print('eloc: ', eloc)
print('len eloc: ', len(eloc))
print('eloc[0]: ', eloc[0])
print('loss_states: ', loss_stats)

# Center the local energy
eloc -= _mean(eloc)
print('energy?: ',eloc[0])

grads = np.empty(
            (n_samples_node, ma.n_par), dtype=np.complex128
        )

for x, eloc_x, grad_x in zip(sam, eloc, grads):
    ma.vector_jacobian_prod(x, eloc_x, grad_x)

    grad1 = _mean(grads, axis=0)
    grad = grad1 / float(batch_size)

    dp = grad
print("grads1: ", grads.shape)
print('dp: ', dp.shape)


loc = np.empty(sam.shape[0:2], dtype=np.complex128)
for i, sample in enumerate(sam):
    _local_values(ha, ma, sample, out=loc[i])
loss_stats = _statistics(loc)

obs_data = tree_map(loss_stats, ha)
print('obs_data: ', obs_data)

#initalize
ma.init_random_parameters(sigma=1)

gs = nk.Vmc(hamiltonian=ha,sampler=sa,optimizer=op,n_samples=1000, n_discard=None,sr=None,)
# print(gs.info())
# gs._forward_and_backward()
# stats = gs._get_mc_stats(ha)
# print(stats)

# start = time.time()
# gs.run(out='RBM', n_iter=600)
# end = time.time()
# runTime = end-start

# import the data from log file
# data = json.load(open("RBM.log"))

# Extract the relevant information
# iters = []
# energy_RBM = []
#
# for iteration in data["Output"]:
#     iters.append(iteration["Iteration"])
#     engTemp = iteration["Energy"]["Mean"]
#     energy_RBM.append(engTemp)
# finalEng = energy_RBM[-1]
# engErr = finalEng - exact_gs_energy
#
#
#
#
#
# # Plot Iteration
# fig, ax1 = plt.subplots()
# plt.title('Central Spin N = 2 ', size=20)
# ax1.plot(iters, energy_RBM - exact_gs_energy, color='red', label='Energy (RBM)')
# ax1.set_ylabel('Energy Error')
# #ax1.set_ylim(0,1.5)
# ax1.set_xlabel('Iteration')
# #plt.axis([0,iters[-1],exact_gs_energy-0.03,exact_gs_energy+0.2])
# plt.show()
#
