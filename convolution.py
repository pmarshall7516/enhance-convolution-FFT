import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve2d

# Direct Convolution Algorithm
def convolution2D(image, kernel):
    # Get dimensions of the image and kernel
    image_height, image_width = len(image), len(image[0])
    kernel_height, kernel_width = len(kernel), len(kernel[0])

    # Calculate the padding needed for valid convolution
    pad_height = kernel_height - 1 
    pad_width = kernel_width - 1

    # Pad the image with zeros
    padded_image = [[0] * (image_width + 2 * pad_width) for _ in range(image_height + 2 * pad_height)]
    for i in range(image_height):
        for j in range(image_width):
            padded_image[i + pad_height][j + pad_width] = image[i][j]      
        
    # Initialize output img
    output = [[0] * (image_width + 1 * pad_width) for _ in range(image_height + 1 * pad_height)]

    # Initialize the output feature map
    output_height = len(output)
    output_width = len(output[0])

    reversed_kernel = [row[::-1] for row in kernel]
    reversed_kernel.reverse()
    
    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Apply the kernel
            for m in range(kernel_height):  # Reverse loop for rows
                for n in range(kernel_width):  # Reverse loop for columns
                    output[i][j] += padded_image[i + m][j + n] * reversed_kernel[m][n]
    
    return output

# FFT-Based Convolution Algorithm
def fft_convolve2d(image, kernel):
    image = np.array(image)
    kernel = np.array(kernel)
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding needed for valid convolution
    pad_height = kernel_height - 1
    pad_width = kernel_width - 1

    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Pad the kernel to match the size of the padded image
    padded_kernel = np.pad(kernel, ((0, image_height - kernel_height), (0, image_width - kernel_width)), mode='constant')

    # Apply FFT to the padded image and kernel
    fft_image = fft2(padded_image)
    fft_kernel = fft2(padded_kernel, s=fft_image.shape)

    # Perform element-wise multiplication in frequency domain
    fft_result = fft_image * fft_kernel

    # Apply inverse FFT to get back to spatial domain
    convolved_result = np.real(ifft2(fft_result))

    # Crop the result to the original size of the image
    convolved_result = convolved_result[pad_height:, pad_width:]

    return convolved_result


#----- Example of Square Image and Kernel -----#
img1 = [[1, 1, 1, 1 ,1], 
        [1, 1, 1, 1, 1],  
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],  
        [1, 1, 1, 1, 1]]

kern1 = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]

# Print Direct Convolution Result
# z = np.array(convolution2D(img1, kern1))
# print("\nConvolved Image (NO FFT): ")
# for line in z:
#     print(line)

# Print FFT-Based Convolution Result
# y = fft_convolve2d(img1, kern1)
# print("\nConvolved Image (FFT): ")
# for line in y:
#     print(line)


#----- Example of Rectangular Image and Kernel ----- #
# img2 = [[1, 1, 1, 1 ,1, 1, 1], 
#         [1, 1, 1, 1 ,1, 1, 1],  
#         [1, 1, 1, 1 ,1, 1, 1],
#         [1, 1, 1, 1 ,1, 1, 1],  
#         [1, 1, 1, 1 ,1, 1, 1]]     

# kern2 = [[1, 1, 1],
#          [1, 1, 1],
#          [1, 1, 1],
#          [1, 1, 1],
#          [1, 1, 1]]

# z = convolution2D(img2,  kern2)
# for line in z:
#     print(line)

# y = fft_convolve2d(img2, kern2)
# print("\nConvolved Image (FFT): ")
# for line in y:
#     print(line)    

# Plot the original input
# plt.subplot(1, 3, 1)
# plt.imshow(img1, cmap='gray')
# plt.title('Original Input')
# plt.colorbar()

# # Plot the Direct Convolution result
# plt.subplot(1, 3, 2)
# plt.imshow(np.abs(z), cmap='gray')
# plt.title('Direct Convolution Result')
# plt.colorbar()

# # Plot the FFT result
# plt.subplot(1, 3, 3)
# plt.imshow(np.abs(y), cmap='gray')
# plt.title('FFT Result')
# plt.colorbar()

# plt.show()

