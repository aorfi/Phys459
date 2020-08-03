from qutip import *
import numpy as np
import cmath

j = complex(0,1)
si = qeye(2)
sx = 0.5 * sigmax()
sy = 0.5 * sigmay()
sz = 0.5 * sigmaz()
sp = sx+j*sy
sm = sx-j*sy

print(1*tensor(2*sz,si,si)+5*tensor(2*sz,2*sz,si)+6*tensor(2*sz,si,2*sz))

print(5*tensor(sp,sm,si,si)+5*tensor(sm,sp,si,si)+6*tensor(sp,si,sm,si)+6*tensor(sm,si,sp,si)+7*tensor(sp,si,si,sm)+7*tensor(sm,si,si,sp)+tensor(si,si,si,si))