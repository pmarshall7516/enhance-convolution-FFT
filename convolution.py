def convolution2D(image, kernel, padding = True):
    # Get dimensions of the image and kernel
    image_height, image_width = len(image), len(image[0])
    kernel_height, kernel_width = len(kernel), len(kernel[0])


    if padding:
        # Calculate the padding needed for valid convolution
        pad_height = kernel_height - 1 
        pad_width = kernel_width - 1
    
        # Pad the image with zeros
        padded_image = [[0] * (image_width + 2 * pad_width) for _ in range(image_height + 2 * pad_height)]
        for i in range(image_height):
            for j in range(image_width):
                padded_image[i + pad_height][j + pad_width] = image[i][j]

        # # Print the padded image to ensure proper padding for kernel
        print("\nPadded Image:")
        for line in padded_image:
            print(line)
    
        # Initialize output img
        output = [[0] * (image_width + 1 * pad_width) for _ in range(image_height + 1 * pad_height)]

    else:
        # No padding necessary. Use original image dimensions
        output = output = [[0] * image_width for _ in range(image_height)]

    # Initialize the output feature map
    output_height = len(output)
    output_width = len(output[0])

    # Test Image Output Initialization
    print("\nOutput Image Initialized:")
    for line in output:
        print(line)
    
    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Apply the kernel
            for m in range(kernel_height):
                for n in range(kernel_width):
                    if padding:
                        output[i][j] += padded_image[i + m][j + n] * kernel[m][n]
                    else:
                        output[i][j] += image[i][j] * kernel[m][n] 
    
    return output


#----- Example of Square Image and Kernel -----#

img1 = [[1, 1, 1, 1 ,1], 
        [1, 1, 1, 1, 1],  
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],  
        [1, 1, 1, 1, 1]]

kern1 = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]

z = convolution2D(img1, kern1)
print("\nConvolved Image: ")
for line in z:
    print(line)



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
# # z = convolution2D(img2,  kern2)
# # for line in z:
# #     print(line)