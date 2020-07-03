from RunTimeInformation import runDescent
import cProfile
import re
import pstats


M = 2
N= 2
B = 1
A0 = 1
cProfile.run('runDescent(N,M,B,A0)', 'restats')
p = pstats.Stats('restats')

p.sort_stats('cumulative').print_stats(10)
