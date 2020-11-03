from qutip import *
import numpy as np
import cmath
import netket as nk

j = complex(0,1)
si = qeye(2)
sx = 0.5 * sigmax()
sy = 0.5 * sigmay()
sz = 0.5 * sigmaz()
sp = sx+j*sy
sm = sx-j*sy

term1 = 4*tensor(sx,sx,si)+4*tensor(sy,sy,si)+4*tensor(sz,sz,si)
term2 = 4*tensor(si,sx,sx)+4*tensor(si,sy,sy)+4*tensor(si,sz,sz)
term3 = 4*tensor(sx,si,sx)+4*tensor(sy,si,sy)+4*tensor(sz,si,sz)
#print(term1+term2+term3)

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

def hei(N):
    sx_list = operatorCreation(N)[0]
    sy_list = operatorCreation(N)[1]
    sz_list = operatorCreation(N)[2]
    H=0
    for n in range(N - 1):
        H += 4*sz_list[n] * sz_list[n + 1] + 4*sx_list[n] * sx_list[n + 1] + 4*sy_list[n] * sy_list[n + 1]
    return H

def heiNetKet(N):
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

    # Iteraction term
    # WHY DO WE NEED THESE SIGNS TO MATCH THE MATRICES?
    itOp = 4*np.kron(sigmaz, sigmaz) +4*np.kron(sigmax, sigmax) + 4*np.kron(sigmay, sigmay)
    #print('ItOp: ', itOp)
    for i in range(N-1):
        operators.append((itOp).tolist())
        sites.append([i, (i+1)])


    # print('sites: ', sites)
    # print('operators: ', operators)
    ha = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    res = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=False)
    print("NetKet Custom ground state energy = {0:.3f}".format(res.eigenvalues[0]))
    #Returns Hamiltonian and Hilbert space
    return ha, hi

# Define a 1d chain
L = 3
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)
hi = nk.hilbert.Spin(s=0.5, graph=g)

ha = nk.operator.Heisenberg(hilbert=hi)
#op = nk.operator.Ising(h=1.321, hilbert=hi, J=0.5)
#print(op.to_dense())
print("NetKet Defined: ", ha.to_dense())
res = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=False)
print("NetKet Defined ground state energy = {0:.3f}".format(res.eigenvalues[0]))

haDefNetKet = heiNetKet(L)[0]
print('Custom Netket: ', haDefNetKet.to_dense())

haDef = hei(L)
print( 'Custom: ' , haDef)
print( 'Custom Energy: ', haDef.groundstate())