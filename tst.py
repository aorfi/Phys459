from pypapi import events, papi_high as high
import numpy

for n in [10, 30, 100, 300, 1000, 10000, 20000]:
    aa=numpy.mgrid[0:n:1,0:n:1][0]
    high.start_counters([events.PAPI_FP_OPS,])
    a=numpy.fft.fft(aa)
    x=high.stop_counters()
    print (n, x) # Or record results in another way