#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from qutip import *
import numpy as np
import scipy
import cmath
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import itertools
from numpy.random import seed
from numpy.random import rand
from itertools import product
from scipy.misc import derivative
from scipy import optimize
from functools import wraps
from time import time
from scipy.optimize import minimize
from matplotlib import gridspec


# In[2]:


#Wrapper to time functions
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ti = time()
        result = f(*args, **kw)
        tf = time()
        t = tf-ti
        return result, t
    return wrap


# In[3]:


#Make basis and get sz values
def operatorCreation(N):
    #operator definitionis
    si = qeye(2)
    sx = 0.5*sigmax()
    sy = 0.5*sigmay()
    sz = 0.5*sigmaz()

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
        
    return sx_list, sy_list, sz_list,id


# In[4]:


#Construct Hamiltonian
def hamiltonian(N,B,A0):
    sx_list = operatorCreation(N)[0]
    sy_list = operatorCreation(N)[1]
    sz_list = operatorCreation(N)[2]
    H = B*sz_list[0] 
    for n in range(N-1):
        H += A0*sz_list[0]*sz_list[n+1] + A0*sx_list[0]*sx_list[n+1] + A0*sy_list[0]*sy_list[n+1]
    return H


# In[5]:


#Check if dengerate
def CheckDegenerate(H):
    estates = H.eigenstates()
    lowState = estates[0][0]
    secLowState = estates[0][1]
    if (abs(lowState-secLowState))<= 1e-10:
        return True
    else:
        return False


# In[6]:


#Get Ground State Energy and Wavefuntion
class GroundState:
    
    def __init__(self, N, B, A0):
        self.hamiltonian = hamiltonian(N, B, A0)
     
    @timing
    def __call__(self):
        #find ground state
        H = self.hamiltonian
        groundState= H.groundstate()
        return groundState[0],groundState[1]


# In[7]:


#Make basis and get sz values
def basisCreation(N):
    sz_list = operatorCreation(N)[2]
    Sbasis = []
    basisState = []
   
    for j in range(2):
        basisState.append(basis(2,j))
    b = itertools.product(basisState,repeat=N)
    basisTensor = list(b)
    #makes Sbasis the correct dimesion of Qobj
    for i in range(2**N):
        c = basisTensor[i][0]
        for j in range(N-1):
            c = tensor(c,basisTensor[i][j+1])
        Sbasis.append(c)

    
    #get sz values for basis states
    sz = np.zeros((2**N,N), dtype = complex)
    a = [[1 for j in range(N)] for i in range(2**N)]
    for i in range(2**N):
        for j in range(N):
            #matrix element <bra|Sz|ket>
            sz[i][j] = sz_list[j].matrix_element(Sbasis[i],Sbasis[i])
    return Sbasis, sz


# In[8]:


#get randomized RBM parameters (between zero and 1)
def ranRBMpar(N,M):
    par = 1-2*np.random.rand(2*(N+M+N*M))
    return par


# In[9]:


#Function to give RBM wavefuntion
def RBM_ansatz(par,N, M):
    Sbasis = basisCreation(N)[0]
    sz = basisCreation(N)[1]
    #make parmeters complex
    num = N+M+N*M
    parC = np.vectorize(complex)(par[:num],par[num:])
    a = parC[:N]
    b = parC[N:N+M]
    W = parC[N+M:].reshape(M,N)
    expTerm = np.zeros(2**N, dtype = complex)
    coshTerm = np.zeros((M,2**N), dtype = complex)
    hidProduct = np.zeros(2**N, dtype = complex)
    psiMValues = np.zeros(2**N, dtype = complex)
    psiM = 0*Sbasis[0]

    for i in range(2**N):
        for m in range(M):
            coshTerm[m][i] = 2*np.cosh(np.dot(W[m],sz[i]) + b[m])
    hidProduct = np.prod(coshTerm, axis = 0) 
    
    for i in range(2**N):
        expTerm[i] = np.exp(np.dot(a,sz[i]))
        psiMValues[i] = expTerm[i]*hidProduct[i]
        psiM += psiMValues[i]*Sbasis[i]
    psiNorm = psiM.unit()
    return psiNorm


# In[10]:


#Variation Energy Definition
def varEnergy(par,N, M, H):
    psiM = RBM_ansatz(par,N, M)
    E = expect(H,psiM)
    norm = psiM.norm()**2
    Enorm = E/norm
    return Enorm


# In[11]:


#Energy Partials
def gradEnergy(par,N, M,H):
    eps = 0.001
    return optimize.approx_fprime(par,varEnergy,eps, N, M,H)


# In[12]:


#Conjugate Gradient Descent
class ConGradDescent:
    
    def __init__(self, N, B, A0):
        H = hamiltonian(N, B, A0)
        isDeg = CheckDegenerate(H)
        if (isDeg == True):
            raise ValueError("Hamiltonian has degenerate ground state")
        else:
            self.hamiltonian = H
     
    @timing 
    def __call__(self, N, M,par):
        H = self.hamiltonian
        min = scipy.optimize.fmin_cg(varEnergy,par,args= (N,M,H),gtol = 1e-04, full_output=True, retall = True, disp=True)
        #Ground State
        found_gs = RBM_ansatz(min[0],N, M)
        found_gs = found_gs.unit()
        #Ground State Energy
        found_gsEnergy =varEnergy(min[0], N, M,H)
        return min, found_gs, found_gsEnergy  


# In[13]:


#Error Calculation
def err(found_gs,gs,found_gsEnergy,gsEnergy):
    engErr = np.abs(found_gsEnergy-gsEnergy)
    waveFunctionErr = found_gs.dag()*gs
    waveFunctionErr = 1-waveFunctionErr.norm()
    
    return engErr,waveFunctionErr


# In[14]:


#Ensures conjugate gradient descent convergance
def CgdConvergance(N, M, B, A0):
    #random parameters
    par = ranRBMpar(N,M)
    #performs conjugate gradient descent
    conGradDescent = ConGradDescent(N, B, A0)
    cgd = conGradDescent(N, M,par)
    #get actual ground state
    groundState = GroundState(N,B,A0)
    ed = groundState()
    #Calculate Error
    error = err(cgd[0][1],ed[0][1],cgd[0][2],ed[0][0])
    relativeErr = np.abs(error[0]/ed[0][0])
    print('relativeErr',relativeErr)
    #Run again if the relative error is larger than 0.01
    runs = 1
    while(relativeErr>0.01):
        par = ranRBMpar(N,M)
        cgd = conGradDescent(N, M,par)
        error = err(cgd[0][1],ed[0][1],cgd[0][2],ed[0][0])
        relativeErr = np.abs(error[0]/ed[0][0])
        runs += 1
        print('relativeErr in Loop',relativeErr)
    return cgd, error, runs


# ## Run Statistics

# In[15]:


# Runs inforamtion is saved at the following locations:
# N=2 M=1 Data/May14/N2M1.json
# N=2 M=2 Data/May14/N2M2.json
# N=2 M=3 Data/May14/N2M3.json
# N=2 M=4 Data/May14/N2M4.json


# In[16]:


# #How to read saved data
# saved = []
# with open('Data/May12/N2M3Bpi.json') as file:
#     for line in file:
#         saved.append(json.loads(line))

# cgdTime,cgdEngErr,cgdStateErr,edTime,len(hisIt)=saved


# ## Generating Data

# In[17]:


#Parameter definition 
N= 2
M=1
B = 1
A0 = 1

cgdResults = [] #gives ground state estimation at each iteration of gd
cgdState = []
cgdTime = []
cgdEngErr = []
cgdStateErr = []

edState = []
edTime = []
ActualEng = []

hisIt = np.arange(5)


# In[65]:


for i in range(len(hisIt)):
    par = ranRBMpar(N,M)
    
    #gradient descent
    conGradDescent = ConGradDescent(N, B, A0)
    cgd = conGradDescent(N,M,par)
    cgdResults.append(cgd)
    cgdState.append(cgd[0])
    cgdTime.append(cgd[1])
    
    
    #exact diagonalization 
    groundState = GroundState(N,B,A0)
    ed = groundState()
    edState.append(ed[0])
    edTime.append(ed[1])
    
    
    #Error
    cgdEngTemp = cgd[0][2]
    edEngTemp = ed[0][0]
    
    ActualEng.append(edEngTemp)
    cgdStateTemp = cgd[0][1]
    edStateTemp = ed[0][1]

    
    cgdErrTemp = err(cgdStateTemp,edStateTemp,cgdEngTemp,edEngTemp)  
    cgdEngErr.append(cgdErrTemp[0])
    cgdStateErr.append(cgdErrTemp[1])
   


# In[66]:


#Save data to JSON file
data = [cgdTime,cgdEngErr,cgdStateErr,edTime,len(hisIt)]
open("TestJSON.json", "w").close()
with open("TestJSON.json", 'a') as file:
    for item in data: 
        line = json.dumps(item)
        file.write(line + '\n')


# In[ ]:





# In[ ]:





# In[70]:





# In[72]:


avEngErr = np.sum(cgdEngErr)/(len(hisIt))
avStateErr = np.sum(cgdStateErr)/(len(hisIt))
avRunTime = np.sum(cgdTime)/(len(hisIt))
print('Average Energy Error: ', avEngErr )
print('Average Ground State Error: ', avStateErr )
print('Average Run Time: ', avRunTime )

cutOff = 0.0001
runsCutOff = sum(i>cutOff for i in cgdEngErr)
prob = (runsCutOff/100)
print("Number of runs abover 0.0001 error:", runsCutOff )
print("Prob of sucess:",  f"{1-prob:.4f}" )
print("Prob of sucess two runs:", f"{1-prob**2:.4f}" )
print("Prob of sucess three runs:", f"{1-prob**3:.4f}" )


