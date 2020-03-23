# Import netket library
import netket as nk
# Import Json, this will be needed to examine log files
import json
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time

# Define a 1d chain
L = 6
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Define the Hilbert space based on this graph
# We impose to have a fixed total magnetization of zero
hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)
# calling the Heisenberg Hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)

# compute the ground-state energy (here we only need the lowest energy, and do not need the eigenstate)
exact_result = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=False)
exact_gs_energy = exact_result.eigenvalues[0]
print('The exact ground-state energy is E0=',exact_gs_energy)

# RBM ansatz with alpha=1
ma = nk.machine.RbmSpin(alpha=5, hilbert=hi)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g)
ma.init_random_parameters(seed=123, sigma=0.01)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)
# Ground State
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.1,
    use_iterative=True,
    method='Gd')

start = time.time()
gs.run(output_prefix='RBM', n_iter=600)
end = time.time()

print('### RBM calculation')
print('Has',ma.n_par,'parameters')
print('The RBM calculation took',end-start,'seconds')

# import the data from log file
data = json.load(open("RBM.log"))

# Extract the relevant information
iters = []
energy_RBM = []

for iteration in data["Output"]:
    iters.append(iteration["Iteration"])
    energy_RBM.append(iteration["Energy"]["Mean"])

fig, ax1 = plt.subplots()
plt.title("Antiferromagnetic Heisenberg Chain")
ax1.plot(iters, energy_RBM, color='red', label='Energy (RBM)')
ax1.set_ylabel('Energy')
ax1.set_xlabel('Iteration')
plt.axis([0, iters[-1], exact_gs_energy - 0.03, exact_gs_energy + 0.2])
plt.axhline(y=exact_gs_energy, xmin=0,
            xmax=iters[-1], linewidth=2, color='k', label='Exact')
ax1.legend()
plt.show()