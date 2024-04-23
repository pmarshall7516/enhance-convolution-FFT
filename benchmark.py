import time
import numpy as np
from scipy.signal import fftconvolve, convolve2d
from convolution import convolution2D, fft_convolve2d
import matplotlib.pyplot as plt

# Function to generate random image and kernel
def generate_random_data(height, width, min_val=1, max_val=50):
    return np.random.randint(min_val, max_val, size=(height, width))

# Function to measure execution time of a function
def measure_execution_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return result, end_time - start_time

# Define image and kernel dimensions
image_sizes = [(1920, 1080), (2560, 1440), (3840, 2160), (3000, 2000), ]
kernel_sizes = [3, 5, 7]

# Results storage
direct_conv_times = []
fft_conv_times = []
image_areas = []

# Iterate over different image and kernel sizes
for size in image_sizes:
    image_width = size[0]
    image_height = size[1]
    for kernel_height in kernel_sizes:
        kernel_width = kernel_height  # Square kernel

    # Generate random image and kernel
    image = generate_random_data(image_height, image_width)
    kernel = generate_random_data(kernel_height, kernel_width)

    # Benchmarking direct convolution
    direct_conv_result, direct_conv_time = measure_execution_time(convolution2D, image, kernel)

    # Benchmarking FFT-based convolution
    fft_conv_result, fft_conv_time = measure_execution_time(fft_convolve2d, image, kernel)

    # Benchmarking scipy's Direct convolution
    #scipy_direct_conv_result, scipy_direct_conv_time = measure_execution_time(convolve2d, image, kernel)

    # Benchmarking scipy's FFT convolution
    #scipy_fft_conv_result, scipy_fft_conv_time = measure_execution_time(fftconvolve, image, kernel)

    direct_conv_times.append(direct_conv_time)
    fft_conv_times.append(fft_conv_time)
    image_areas.append(image_width*image_height)

# print("Image:")
# print(image)

# print("Kernel:")
# print(kernel)


# print("Direct Convolution Result:")
# for line in direct_conv_result:
#     print(line)
# print("Direct Convolution Execution Time:", direct_conv_time)

# print("FFT Convolution Result:")
# print(fft_conv_result)
# print("FFT-Based Convolution Execution Time:", fft_conv_time)

# print("Scipy Direct Convolution Result:")
# print(scipy_direct_conv_result)
# print("Scipy's FFT Convolution Execution Time:", scipy_direct_conv_time)

# print("Scipy FFT-Convolution Result:")
# print(scipy_fft_conv_result)
# print("Scipy's FFT Convolution Execution Time:", scipy_fft_conv_time)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(image_areas, direct_conv_times, label='Direct Convolution')
plt.plot(image_areas, fft_conv_times, label='FFT Convolution')
plt.title('Execution Time vs. Image Size')
plt.xlabel('Image Size (Width)')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()


