# Imports
from scipy import signal
import time
import numpy as np

signal_lengths = [100, 1000, 10_000, 100_000, 1_000_000]

for length in signal_lengths:

    a = np.random.rand(length)  
    b = np.random.rand(length)

    # Measure execution time for signal.convolve
    start_time_convolve = time.time()
    y = signal.convolve(a, b)
    end_time_convolve = time.time()
    time_taken_convolve = end_time_convolve - start_time_convolve

    # Measure execution time for signal.fftconvolve
    start_time_fftconvolve = time.time()
    z = signal.fftconvolve(a, b)
    end_time_fftconvolve = time.time()
    time_taken_fftconvolve = end_time_fftconvolve - start_time_fftconvolve

    # Print Signal Length
    print('SIGNALS OF LENGTH: ', length)

    # Print convoluted sequences
    print('Convoluted sequence using signal.convolve:', y)
    print('Convoluted sequence using signal.fftconvolve:', z)

    # Print time difference
    print('Time taken for signal.convolve:', time_taken_convolve)
    print('Time taken for signal.fftconvolve:', time_taken_fftconvolve)
