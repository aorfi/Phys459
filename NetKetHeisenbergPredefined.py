import netket as nk
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import time
import json
import multiprocessing as mp
import os


def hamiltonian(N, J):
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


class ExactDigonalization:
    def __init__(self,N,J):
        self.ha,self.hi = hamiltonian(N,J)
        self.N = N

    def __call__(self):
        haMatrix = self.ha.to_dense()
        e, v = np.linalg.eigh(haMatrix)
        inds = np.argsort(e)
        e = e[inds]
        v = v[:, inds]
        return e,v

class RBM:
    def __init__(self,N,J,alpha):
        self.ha,self.hi = hamiltonian(N,J)
        # Define machine
        self.ma = nk.machine.RbmSpin(alpha = alpha, hilbert= self.hi)
        # Define sampler
        self.sa = nk.sampler.MetropolisLocal(machine=self.ma)
        # Optimizer
        self.op = nk.optimizer.Sgd(learning_rate=0.05)

    def __call__(self):
        # Initialize parameters
        self.ma.init_random_parameters(sigma=1)
        print('RBM Parameters = ', self.ma.parameters)

        # Stochastic reconfiguration
        # gs = nk.variational.Vmc(
        #     hamiltonian=self.ha,
        #     sampler=self.sa,
        #     optimizer=self.op,
        #     n_samples=1000,
        #     diag_shift=0.1,
        #     use_iterative=True,
        #     method='Gd')

        gs = nk.Vmc(
            hamiltonian=self.ha,
            sampler=self.sa,
            optimizer=self.op,
            n_samples=1,
            n_discard=None,
            sr=None,
        )

        start = time.time()
        gs.run(output_prefix='RBM', n_iter=600)
        end = time.time()
        runTime = end-start
        return runTime

def runDescent(N,J,alpha):
    rbm = RBM(N, J, alpha)
    runTime = rbm()

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






J=1
N = 2
alpha = 1

ha,hi = hamiltonian(N, J)

exact = ExactDigonalization(N,J)
evalues, evectors = exact()
exact_gs_energy = evalues[0]
print("ground state energy = ", exact_gs_energy)
print('eigen values = ', evalues)

print(hi.size)
print("Hamiltonian= \n", ha.to_dense())



# Histogram
hisIt = np.arange(50)
engErr = []
runTime = []

for i in range(len(hisIt)):
    runtimeTemp, engErrTemp = runDescent(N,J,alpha)
    runTime.append(runtimeTemp)
    engErr.append(engErrTemp)

#Save data to JSON file
data = [engErr,runTime]
open("Data/06-15-20/HeiN2MetLocal.json", "w").close()
with open('Data/06-15-20/HeiN2MetLocal.json', 'a') as file:
    for item in data:
        line = json.dumps(item)
        file.write(line + '\n')


# One Run
rbm = RBM(N, J, alpha)
runTime = rbm()

# import the data from log file
data = json.load(open("RBM.log"))

# Extract the relevant information
iters = []
energy_RBM = []
engErr = []



for iteration in data["Output"]:
    iters.append(iteration["Iteration"])
    engTemp = iteration["Energy"]["Mean"]
    energy_RBM.append(engTemp)
    finalEng = energy_RBM[-1]
    engErrTemp = finalEng - exact_gs_energy
    engErr.append(engErrTemp)

print('Final Energy: ', energy_RBM[-1])
fig, ax1 = plt.subplots()
plt.title('Heisenberg Spin Chain N = 2', size=20)
ax1.plot(iters, engErr, color='red', label='Energy (RBM)')
ax1.set_ylabel('Energy Error')
#ax1.set_ylim(0,1.5)
ax1.set_xlabel('Iteration')
#plt.axis([0,iters[-1],exact_gs_energy-0.03,exact_gs_energy+0.2])
plt.show()
