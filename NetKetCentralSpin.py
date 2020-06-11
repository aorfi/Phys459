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
    g = nk.graph.Edgeless(N)
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

    def __call__(self):
        res = nk.exact.full_ed(self.ha, first_n=1)
        return res.eigenvalues, res.eigenvectors


class RBM:
    def __init__(self,N,B,A,alpha):
        self.ha,self.hi = hamiltonian(N,B,A)
        # Define machine
        self.ma = nk.machine.RbmSpin(alpha = alpha, hilbert= self.hi)
        # Define sampler
        self.sa = nk.sampler.MetropolisExchange(machine=self.ma)
        # Optimizer
        self.op = nk.optimizer.Sgd(learning_rate=0.05)

    def __call__(self):
        # Initialize parameters
        self.ma.init_random_parameters(seed=123, sigma=0.01)

        # Stochastic reconfiguration
        gs = nk.variational.Vmc(
            hamiltonian=self.ha,
            sampler=self.sa,
            optimizer=self.op,
            n_samples=1000,
            diag_shift=0.1,
            use_iterative=True,
            method='Gd')

        start = time.time()
        gs.run(output_prefix='RBM', n_iter=600)
        end = time.time()
        runTime = end-start
        return runTime

def runDescent(N,alpha,B,A):
    rbm = RBM(N, B, A, alpha)
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



B=1
A=1
N = 2
alpha = 1 # (M=2)


exact = ExactDigonalization(N,B,A)
evalues, evectors = exact()
exact_gs_energy = evalues[0]
print("ground state energy = ", exact_gs_energy)


# hisIt = np.arange(50)
# engErr = []
# runTime = []
#
# for i in range(len(hisIt)):
#     runtimeTemp, engErrTemp = runDescent(N,alpha,B,A)
#     runTime.append(runtimeTemp)
#     engErr.append(engErrTemp)


# #Save data to JSON file
# data = [engErr,runTime]
# open("Data/06-11-20/NetKetN2M2.json", "w").close()
# with open('Data/06-11-20/NetKetN2M2.json', 'a') as file:
#     for item in data:
#         line = json.dumps(item)
#         file.write(line + '\n')


rbm = RBM(N, B, A, alpha)
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


print(engErr)
print(runTime)
#Save data to JSON file
# data = [engErr,runTime]
# open("Data/06-11-20/NetKetN2M2.json", "w").close()
# with open('Data/06-11-20/NetKetN2M2.json', 'a') as file:
#     for item in data:
#         line = json.dumps(item)
#         file.write(line + '\n')


# print('RBM Ground State Energy :', finalEng)
#
#
fig, ax1 = plt.subplots()
plt.title('Central Spin', size=20)
ax1.plot(iters, engErr, color='red', label='Energy (RBM)')

ax1.set_ylabel('Energy Error')
ax1.set_ylim(0,1.5)
ax1.set_xlabel('Iteration')
#plt.axis([0,iters[-1],exact_gs_energy-0.03,exact_gs_energy+0.2])

plt.show()

