#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


import multiprocessing as mp
import os


# In[ ]:


def cube(x):
    return x**3

ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
pool = mp.Pool(processes=ncpus)
data = [1, 2, 3, 4, 5, 6]
results = [pool.apply_async(cube, args=(x,)) for x in data]
cubes = [p.get() for p in results]
print(cubes)


# In[ ]:


#Save data to JSON file
open("TestJSON.json", "w").close()
with open('TestJSON.json', 'a') as file:
    for item in cubes: 
        line = json.dumps(item)
        file.write(line + '\n')


# In[ ]:




