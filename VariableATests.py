import netket as nk
import json
import os
from qutip import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib import gridspec
import itertools
from numpy.random import seed
from scipy import optimize
from functools import wraps
import time

import copy
import pickle
from pickle import load, dump
import collections
from collections import OrderedDict
import multiprocess as mp
#from multiprocess.pool import ThreadPool as Pool
import os


# Wrapper to time functions
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ti = time.time()
        result = f(*args, **kw)
        tf = time.time()
        t = tf - ti
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
def hamiltonian(N, B, A0):
    sx_list = operatorCreation(N)[0]
    sy_list = operatorCreation(N)[1]
    sz_list = operatorCreation(N)[2]
    H = B * sz_list[0]
    for n in range(N - 1):
        H += A0 * sz_list[0] * sz_list[n + 1] + A0 * sx_list[0] * sx_list[n + 1] + A0 * sy_list[0] * sy_list[n + 1]
    return H

def hamiltonianA(N, B, A, N0):
    sx_list = operatorCreation(N)[0]
    sy_list = operatorCreation(N)[1]
    sz_list = operatorCreation(N)[2]
    H = B * sz_list[0]

    for n in range(N - 1):
        Ak = A/(N0)*np.exp(-n/N0)
        H += Ak  * sz_list[0] * sz_list[n + 1] + Ak  * sx_list[0] * sx_list[n + 1] + Ak  * sy_list[0] * sy_list[n + 1]
    return H

def hamiltonianNetKet(N, B, A, N0):
    # Make graph with no edges of length N
    #g = nk.graph.Edgeless(N)
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    # Define sigma matrices
    sigmaz = -0.5 * np.array([[1, 0], [0, -1]])
    sigmax = 0.5 * np.array([[0, 1], [1, 0]])
    sigmay = -0.5 * np.array([[0, -1j], [1j, 0]])
    operators = []
    sites = []
    # Central spin term
    operators.append((B * sigmaz).tolist())
    sites.append([0])
    # Iteraction term
    itOp = np.kron(sigmaz, sigmaz) + np.kron(sigmax, sigmax) + np.kron(sigmay, sigmay)
    for i in range(N - 1):
        Ak = A / (N0) * np.exp(-i / N0)
        operators.append((Ak * itOp).tolist())
        sites.append([0, (i+1)])
    print('sites: ', sites)
    print('operators: ', operators)
    ha = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    res = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=False)
    print("NetLEt ground state energy = {0:.3f}".format(res.eigenvalues[0]))
    #Returns Hamiltonian and Hilbert space
    return ha, hi

N = 6
B = 1
A = 1
N0 = N/2
H = hamiltonianA(N,B,1,N0)
idx = [3, 2, 1, 0]
Hflip = np.flip(H.full())
#print(Hflip)

haSR, hiSR = hamiltonianNetKet(N, B, A, N0)
#print(haSR.to_dense())
sub = haSR.to_dense()-Hflip
print(sub.nonzero())