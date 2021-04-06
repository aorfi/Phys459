# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json

def CSHam(N, B, Ak):
    # Make graph with of length N with no periodic boundary conditions
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hilbertSpace = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
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
    hamiltonian = nk.operator.LocalOperator(hilbertSpace, operators=operators, acting_on=sites, dtype=complex)
    return hamiltonian, hilbertSpace

def exactDiagonalization(hamiltonian):
    # Changes Hamiltonian to matrix form
    haMatrix = hamiltonian.to_dense()
    # Gets eigenvalues and vectors
    eigenValues, v = np.linalg.eigh(haMatrix)
    # Orders from smallest to largest
    eigenVectors = [v[:, i] for i in range(len(eigenValues))]
    return eigenValues, eigenVectors

# Error Calculation (Input: the found state, the state from exact diagonalization, the found energy, the energy from exact diagonalization)
def err(state, edState, eng, edEng,N):
    engErr = np.abs(eng - edEng)
    overlap = np.dot(state.conj().reshape(2**N, 1).T, edState.reshape(2**N, 1))
    waveFunctionErr = 1 - np.linalg.norm(overlap)
    return engErr, waveFunctionErr

# 1D Lattice
N = 2
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
print('Ham ', ha.to_dense())

N = 2
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
ha, hi = CSHam(N,B,Ak)
print('CSHam ', ha.to_dense())


# RBM Spin Machine
ma = nk.models.RBM(alpha=1, dtype=complex,use_visible_bias=True, use_hidden_bias=True)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)

# SR
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
vs = nk.variational.MCState(sa, ma, n_samples=1000, n_discard=100)
vs.init_parameters(nk.nn.initializers.normal(stddev=0.25))
print(vs.to_array())

# Variational monte carlo driver with a variational state
gs = nk.VMC(ha, op, variational_state=vs, sr=sr)

# Run the optimization for 300 iterations
gs.run(n_iter=300, out='test')
e,v = exactDiagonalization(ha)



data = json.load(open('test.log'))
# Extract the relevant information
iters = data["Energy"]["iters"]
energy_RBM = data["Energy"]["Mean"]['real']
print(energy_RBM )
print('ed energy ',e[0])
print('ed state', v[0])
print('final state', vs.to_array())
print('final eng ', energy_RBM[-1])



engErr, stateErr = err(vs.to_array(), v[0], energy_RBM[-1] , e[0],N)
print('eng Error ', engErr)
print('state Error ', stateErr)


cutoff_eng = []
for i in energy_RBM:
    if i != None:
        cutoff_eng.append(i- e[0])

plt.plot(np.arange(len(cutoff_eng)),cutoff_eng)
plt.title("Heisenberg N=2", size=15)
plt.ylabel("Eng Error", size=10)
plt.xlabel("iteration", size=10)
plt.show()