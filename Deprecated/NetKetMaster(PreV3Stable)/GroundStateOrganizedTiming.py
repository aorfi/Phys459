# Activate the qutip-env-linux environment (source qutip-env-linux/bin/activate)
# If running on the cluster then you also need mpi4py which is preloaded on Beluga (module load mpi4py)
import netket as nk
import json
from qutip import *
import numpy as np
import time
import multiprocess as mp
from collections import OrderedDict
from pickle import dump
import os
import matplotlib.pyplot as plt
import scipy
from matplotlib import gridspec
plt.style.use('seaborn')
from scipy.stats import norm
import math

import netket as _nk

from netket.operator import local_values as _local_values
from netket.stats import (
    statistics as _statistics,
    mean as _mean,
    sum_inplace as _sum_inplace,
)

from netket.vmc_common import info, tree_map
from netket.abstract_variational_driver import AbstractVariationalDriver

fbTime = []
samplesTime = []
engTime = []
dpTime = []

class VmcLocal(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self,
        hamiltonian,
        sampler,
        optimizer,
        n_samples,
        n_discard=None,
        sr=None,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian (AbstractOperator): The Hamiltonian of the system.
            sampler: The Monte Carlo sampler.
            optimizer (AbstractOptimizer): Determines how optimization steps are performed given the
                bare energy gradient.
            n_samples (int): Number of Markov Chain Monte Carlo sweeps to be
                performed at each step of the optimization.
            n_discard (int, optional): Number of sweeps to be discarded at the
                beginning of the sampling, at each step of the optimization.
                Defaults to 10% of the number of samples allocated to each MPI node.
            sr (SR, optional): Determines whether and how stochastic reconfiguration
                is applied to the bare energy gradient before performing applying
                the optimizer. If this parameter is not passed or None, SR is not used.

        Example:
            Optimizing a 1D wavefunction with Variational Monte Carlo.
            #
            # >>> import netket as nk
            # >>> SEED = 3141592
            # >>> g = nk.graph.Hypercube(length=8, n_dim=1)
            # >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            # >>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
            # >>> ma.init_random_parameters(seed=SEED, sigma=0.01)
            # >>> ha = nk.operator.Ising(hi, h=1.0)
            # >>> sa = nk.sampler.MetropolisLocal(machine=ma)
            # >>> op = nk.optimizer.Sgd(learning_rate=0.1)
            # >>> vmc = nk.Vmc(ha, sa, op, 200)

        """
        super(VmcLocal, self).__init__(
            sampler.machine, optimizer, minimized_quantity_name="Energy"
        )

        self._ham = hamiltonian
        self._sampler = sampler
        self.sr = sr
        print("init")
        self.count = 0

        self._npar = self._machine.n_par

        self._batch_size = sampler.sample_shape[0]

        # Check how many parallel nodes we are running on
        self.n_nodes = _nk.utils.n_nodes

        self.n_samples = n_samples
        self.n_discard = n_discard

        self._dp = None

    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, sr):
        self._sr = sr
        if self._sr is not None:
            self._sr.setup(self.machine)

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n_samples):
        if n_samples <= 0:
            raise ValueError(
                "Invalid number of samples: n_samples={}".format(n_samples)
            )

        n_samples_chain = int(math.ceil((n_samples / self._batch_size)))
        self._n_samples_node = int(math.ceil(n_samples_chain / self.n_nodes))

        self._n_samples = int(self._n_samples_node * self._batch_size * self.n_nodes)

        self._samples = None

        self._grads = None
        self._jac = None

    @property
    def n_discard(self):
        return self._n_discard

    @n_discard.setter
    def n_discard(self, n_discard):
        if n_discard is not None and n_discard < 0:
            raise ValueError(
                "Invalid number of discarded samples: n_discard={}".format(n_discard)
            )
        self._n_discard = (
            int(n_discard)
            if n_discard != None
            else self._n_samples_node * self._batch_size // 10
        )

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """
        start = time.time()
        self._sampler.reset()

        # Burnout phase
        self._sampler.generate_samples(self._n_discard)

        startSamples = time.time()
        # Generate samples and store them
        self._samples = self._sampler.generate_samples(
            self._n_samples_node, samples=self._samples
        )
        endSamples = time.time()
        samplesTime.append(endSamples-startSamples)

        startEng = time.time()
        # Compute the local energy estimator and average Energy
        eloc, self._loss_stats = self._get_mc_stats(self._ham)
        self.count +=1
        print(self.count)
        # Center the local energy
        eloc -= _mean(eloc)


        samples_r = self._samples.reshape((-1, self._samples.shape[-1]))
        eloc_r = eloc.reshape(-1, 1)
        endEng = time.time()
        engTime.append(endEng - startEng)

        # Perform update
        startDP = time.time()
        if self._sr:
            if self._sr.onthefly:

                self._grads = self._machine.vector_jacobian_prod(
                    samples_r, eloc_r / self._n_samples, self._grads
                )

                self._grads = tree_map(_sum_inplace, self._grads)

                self._dp = self._sr.compute_update_onthefly(
                    samples_r, self._grads, self._dp
                )

            else:
                # When using the SR (Natural gradient) we need to have the full jacobian
                self._grads, self._jac = self._machine.vector_jacobian_prod(
                    samples_r,
                    eloc_r / self._n_samples,
                    self._grads,
                    return_jacobian=True,
                )

                self._grads = tree_map(_sum_inplace, self._grads)

                self._dp = self._sr.compute_update(self._jac, self._grads, self._dp)

        else:
            # Computing updates using the simple gradient
            self._grads = self._machine.vector_jacobian_prod(
                samples_r, eloc_r / self._n_samples, self._grads
            )

            self._grads = tree_map(_sum_inplace, self._grads)

            #  if Real pars but complex gradient, take only real part
            # not necessary for SR because sr already does it.
            if not self._machine.has_complex_parameters:
                self._dp = tree_map(lambda x: x.real, self._grads)
            else:
                self._dp = self._grads
        endDP = time.time()
        dpTime.append(endDP - startDP)

        end = time.time()
        runTime = end - start
        fbTime.append(runTime)
        return self._dp

    @property
    def energy(self):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def _estimate_stats(self, obs):
        if self._samples is None:
            raise RuntimeError(
                "Vmc driver needs to perform a step before .estimate() can be "
                "called. To get VMC estimates outside of optimization, use "
                "netket.variational.estimate_expectations instead."
            )
        return self._get_mc_stats(obs)[1]

    def reset(self):
        self._sampler.reset()
        super().reset()

    def _get_mc_stats(self, op):

        samples_r = self._samples.reshape((-1, self._samples.shape[-1]))

        loc = _local_values(op, self._machine, samples_r).reshape(
            self._samples.shape[0:2]
        )

        # notice that loc.T is passed to statistics, since that function assumes
        # that the first index is the batch index.
        return loc, _statistics(loc.T)

    def __repr__(self):
        return "Vmc(step_count={}, n_samples={}, n_discard={})".format(
            self.step_count, self.n_samples, self.n_discard
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian ", self._ham),
                ("Machine     ", self._machine),
                ("Optimizer   ", self._optimizer),
                ("SR solver   ", self._sr),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)




# Central Spin Hamiltonian and Hilbert space, inputs are Hamiltonian parameters
def CSHam(N, B, Ak):
    # Make graph with of length N with no periodic boundary conditions
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hilbertSpace = nk.hilbert.Spin(s=0.5, graph=g)
    # Define pauli matrices
    pauliz = 0.5 * np.array([[1, 0], [0, -1]])
    paulix = 0.5 * np.array([[0, 1], [1, 0]])
    pauliy = 0.5 * np.array([[0, -1j], [1j, 0]])
    operators = []
    sites = []
    # Central spin term
    operators.append((B * pauliz).tolist())
    sites.append([0])
    # Interaction term
    itOp = np.kron(pauliz, pauliz) + np.kron(paulix, paulix) + np.kron(pauliy, pauliy)
    for i in range(N - 1):
        operators.append((Ak[i] * itOp).tolist())
        sites.append([0, (i+1)])
    # Create hamiltonian
    hamiltonian = nk.operator.LocalOperator(hilbertSpace, operators=operators, acting_on=sites)
    return hamiltonian, hilbertSpace

# Heisenberg  Hamiltonian and Hilbert space, inputs are Hamiltonian parameters
def heiHam(N, J, h):
    # Make graph with of length N with no periodic boundary conditions
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hilbertSpace = nk.hilbert.Spin(s=0.5, graph=g)
    # Define pauli matrices
    pauliz = 0.5 * np.array([[1, 0], [0, -1]])
    paulix = 0.5 * np.array([[0, 1], [1, 0]])
    pauliy = 0.5 * np.array([[0, -1j], [1j, 0]])
    operators = []
    sites = []
    # Interaction term
    itOp = J*np.kron(pauliz, pauliz) + J*np.kron(paulix, paulix) + J*np.kron(pauliy, pauliy)
    for i in range(N-1):
        operators.append((itOp).tolist())
        sites.append([i, (i+1)])
    # Field Term
    for i in range(N):
        operators.append((h*sigmaz).tolist())
        sites.append([i])
    hamiltonian = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    return hamiltonian, hilbertSpace

# Exact Diagonalization, returns eigenvalues and vectors, inputs Hamiltonian
def exactDiagonalization(hamiltonian):
    # Changes Hamiltonian to matrix form
    haMatrix = hamiltonian.to_dense()
    # Gets eigenvalues and vectors
    eigenValues, v = np.linalg.eigh(haMatrix)
    # Orders from smallest to largest
    eigenVectors = [v[:, i] for i in range(len(eigenValues))]
    return eigenValues, eigenVectors

# NetKet RBM with stochastic reconfiguration descent
class RBM:
    def __init__(self, N, hamiltonian, hilbertSpace, machine):
        # Assign inputsv[:, i]
        self.hamiltonian, self.hilbertSpace, self.machine, self.N = hamiltonian, hilbertSpace, machine, N
        # Define sampler
        self.sampler = nk.sampler.MetropolisLocal(machine=self.machine)
        # Define optimizer
        self.optimizer = nk.optimizer.Sgd(self.machine,learning_rate=0.05)

    # Output is the name of the output file in which the descent data is stored
    def __call__(self, output):
        gs = VmcLocal(hamiltonian=self.hamiltonian,
                                sampler=self.sampler,
                                optimizer=self.optimizer,
                                # Number of samples use in the MC estimation
                                n_samples=1000,
                                sr = nk.optimizer.SR(self.machine))
        # Start timing
        start = time.time()
        # Set the output files as well as number of iterations in the descent
        gs.run(out=output, n_iter=1000)
        end = time.time()
        runTime = end - start
        # Import the data from log file
        data = json.load(open(output+'.log'))
        # Extract the relevant information
        iters = []
        energy_RBM = []
        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            engTemp = iteration["Energy"]["Mean"]
            energy_RBM.append(engTemp)
        finalEng = energy_RBM[-1]
        # Get machine statethe state of the machine as an array
        state = self.machine.to_array()
        # Outputs the final energy, the final state, and the runtime
        return finalEng, state, runTime

# Error Calculation (Input: the found state, the state from exact diagonalization, the found energy, the energy from exact diagonalization)
def err(state, edState, eng, edEng,N):
    engErr = np.abs(eng - edEng)
    overlap = np.dot(state.conj().reshape(2**N, 1).T, edState.reshape(2**N, 1))
    waveFunctionErr = 1 - np.linalg.norm(overlap)
    return engErr, waveFunctionErr

# Combines all steps into a function to run on the cluster
def runDescentCS(N,B,Ak,alpha):
    # Define hamiltonian and hibert space (need to do this here cause can't use netket objects as input to use multiprocessing functions)
    ha, hi = CSHam(N,B,Ak)
    # Define machine
    ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=True, use_hidden_bias=True)
    # Initialize the RBM parameters (Gaussian with sd 0.25)
    ma.init_random_parameters(seed=None, sigma=0.25)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma)
    # Run RBM
    eng, state, runTime = rbm("Logs/CS"+str(N))
    return eng, state, runTime

def parameterOutputList(machine):
    paramsComplex = machine.parameters
    parameters = []
    for i in paramsComplex:
        parameters.append(i.real)
        parameters.append(i.imag)
    return parameters

# Initializes random RBM parameters
def ranPar(N, alpha, ma):
    M = alpha * N
    np.random.RandomState()
    par = 0.25*np.random.randn(2 * (N + M + N * M))
    # Change to a,b,w
    num = N + M + N * M
    parC = np.vectorize(complex)(par[:num], par[num:])
    a = parC[:N]
    b = parC[N:N + M]
    w = parC[N + M:].reshape(M, N)
    w = np.array(w).T
    rbmOrderedDict = OrderedDict([('a', a), ('b', b), ('w', w)])
    # Save parameters so they can be loaded into the netket machine
    with open("Logs/par" + str(par[0]) + ".json", "wb") as output:
        dump(rbmOrderedDict, output)
    # Load into ma
    ma.load("Logs/par" + str(par[0]) + ".json")
    return par

for i in range(12):
    # Hamiltonian Parameters
    N = 2+i
    #B = 1
    B=N/2
    A = N/2
    alpha = 1
    M = alpha*N
    N0 = N/2
    # List of Ak
    Ak = []
    for i in range(N - 1):
        Ak_i = A / (N0) * np.exp(-i / N0)
        #Ak_i = 1
        Ak.append(Ak_i)
    # Define hamiltonian and hilbert space
    ha, hi = CSHam(N,B,Ak)

    # # Exact Diagonalization
    e, v = exactDiagonalization(ha)
    #Ground state energy
    edEng = e[0]
    # Ground state
    edState = v[0]

    # Lists for Histogram Data
    numRuns = 1
    hisIt = np.arange(numRuns)
    engErr = []
    stateErr = []
    runTime = []

    # # Node Information
    # ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=50))
    # pool = mp.Pool(processes=ncpus)
    # # Run Descent
    # resultsSR = [pool.apply(runDescentCS, args=(N,B,Ak,alpha)) for x in hisIt]

    # Get errors for each run in histogram
    for i in range(len(hisIt)):
        print("here")
        engTemp, stateTemp, runTimeTemp = runDescentCS(N,B,Ak,alpha)
        runTime.append(runTimeTemp)
        errSR = err(np.asmatrix(stateTemp), edState, engTemp, edEng,N)
        engErr.append(errSR[0])
        stateErr.append(errSR[1])
    print('Eng error ', engErr)
    print('State error ', stateErr)


    # #Save data to JSON file
    # data = [engErr, stateErr, runTime]
    # fileName = "Data/21-02-16/varN"+str(N)+"M" + str(M)+".json"
    # open(fileName, "w").close()
    # with open(fileName, 'a') as file:
    #     for item in data:
    #         line = json.dumps(item)
    #         file.write(line + '\n')
    # print('SAVED')
    #
    #
    #Save data to JSON file

    print(fbTime)
    print(len(fbTime))

    #Save data to JSON file
    data = [fbTime, samplesTime, engTime, dpTime]
    fileName = "Data/21-03-23/timeN"+str(N)+"M" + str(M)+".json"
    open(fileName, "w").close()
    with open(fileName, 'a') as file:
        for item in data:
            line = json.dumps(item)
            file.write(line + '\n')
    print('SAVED')
    fbTime = []
    samplesTime = []
    engTime = []
    dpTime = []
