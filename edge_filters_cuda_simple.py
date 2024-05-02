from numba import cuda, void, float32, uint8, int32
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from time import perf_counter_ns
import pandas as pd


@cuda.jit(void(float32[:,:], uint8[:, :], int32[:, :], int32[:, :]))
def dfilter(input_image, output_image, kernel_x, kernel_y):
    
    i, j = cuda.grid(2)
    height, width = input_image.shape
    kernel_height, kernel_width = kernel_x.shape

    # Calculate the margins to avoid boundary issues
    margin_y, margin_x = kernel_height // 2, kernel_width // 2

    if (margin_y <= i < height - margin_y) and (margin_x <= j < width - margin_x):
        Gx, Gy = 0, 0

        # Apply operator dynamically based on kernel size
        for u in range(-margin_y, margin_y + 1):
            for v in range(-margin_x, margin_x + 1):
                img_val = input_image[i + u, j + v]
                Gx += kernel_x[u + margin_y, v + margin_x] * img_val
                Gy += kernel_y[u + margin_y, v + margin_x] * img_val

        # The magnitude of the gradient
        output_image[i, j] = min(255, max(0, math.sqrt(Gx**2 + Gy**2)))


@cuda.jit(void(float32[:,:], uint8[:, :], int32[:, :]))
def gfilter(input_image, blurred_image, Gausskernel):
    
    i, j = cuda.grid(2)
    height, width = input_image.shape
    kernel_height, kernel_width = Gausskernel.shape

    # Calculate the margins to avoid boundary issues
    margin_y, margin_x = kernel_height // 2, kernel_width // 2

    if (margin_y <= i < height - margin_y) and (margin_x <= j < width - margin_x):
        Gx, Gy = 0, 0

        # Apply operator dynamically based on kernel size
        for u in range(-margin_y, margin_y + 1):
            for v in range(-margin_x, margin_x + 1):
                img_val = input_image[i + u, j + v]
                sum_val = 0
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        sum_val += Gausskernel[a+1, b+1] * input_image[i+a, j+b]
                blurred_image[i, j] = sum_val

        # The magnitude of the gradient
        blurred_image[i, j] = min(255, max(0, math.sqrt(Gx**2 + Gy**2)))

@cuda.jit(void(float32[:,:], uint8[:, :], int32[:, :], int32[:, :], uint8[:, :]))
def opfilter(input_image, output_image, kernel, Gausskernel, blurred_image):
    i, j = cuda.grid(2)
    height, width = input_image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the margins to avoid boundary issues
    margin_y, margin_x = kernel_height // 2, kernel_width // 2

    if (margin_y <= i < height - margin_y) and (margin_x <= j < width - margin_x):
        Gx, Gy = 0, 0

        # Apply operator dynamically based on kernel size
        for u in range(-margin_y, margin_y + 1):
            for v in range(-margin_x, margin_x + 1):
                img_val = blurred_image[i + u, j + v]
                Gx += kernel[u + margin_y, v + margin_x] * img_val
                Gy += kernel[u + margin_y, v + margin_x] * img_val

        # The magnitude of the gradient
        output_image[i, j] = min(255, max(0, math.sqrt(Gx**2 + Gy**2)))
        

@cuda.jit(void(float32[:,:], uint8[:, :], int32[:, :], int32[:, :]))
def cfilter(blurred_image, output_image, kernel_x, kernel_y):
    i, j = cuda.grid(2)
    height, width = blurred_image.shape
    kernel_height, kernel_width = kernel_x.shape
    
    # Calculate the margins to avoid boundary issues
    margin_y, margin_x = kernel_height // 2, kernel_width // 2
    
    if (margin_y <= i < height - margin_y) and (margin_x <= j < width - margin_x):
        Gx, Gy = 0, 0
    
        # Apply Sobel operator dynamically based on kernel size
        for u in range(-margin_y, margin_y + 1):
            for v in range(-margin_x, margin_x + 1):
                img_val = blurred_image[i + u, j + v]
                Gx += kernel_x[u + margin_y, v + margin_x] * img_val
                Gy += kernel_y[u + margin_y, v + margin_x] * img_val
    
        # The magnitude of the gradient
        output_image[i, j] = min(255, max(0, math.sqrt(Gx**2 + Gy**2)))   
        
        
def main():
    # Read the image using matplotlib
    print("Running sobel filter using cuda")
    # Load image
    image_path = 'images/im1.png'  # Update with your image path
    input_image = mpimg.imread(image_path)

    # Check if the image is in color (3 channels)
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        # Convert to grayscale using np.dot
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])
 
    input_image = input_image.astype(np.float32)

    # Define block size and grid size
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(input_image.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(input_image.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Define Sobel kernels
    # SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32)
    SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32)
    
    
    # Define Prewitt kernels

    PREWITT_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32)
    PREWITT_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32)
    
    # Define Scharr kernels
    SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.int32)
    SCHARR_Y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.int32)
    
    #Gaussian Kernel
    Gausskernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.int32) / 16
    
    #Laplace Kernel
    Laplacekernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.int32)
    
    # canny operator kernels
    CANNY_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    CANNY_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])




    filtered_image = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    blurred_image = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)


    def sobel_filter_helper():
        nonlocal filtered_image
        dfilter[blockspergrid, threadsperblock](input_image, filtered_image, SOBEL_X, SOBEL_Y)
        cuda.synchronize()

    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        sobel_filter_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    

    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Sobel Filter Result')
    plt.axis('off')
    plt.show()
    
    print("Running prewitt filter using cuda")
    def prewitt_filter_helper():
        nonlocal filtered_image
        dfilter[blockspergrid, threadsperblock](input_image, filtered_image, PREWITT_X, PREWITT_Y)
        cuda.synchronize()

    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        prewitt_filter_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    

    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('prewitt Filter Result')
    plt.axis('off')
    plt.show()

    print("Running scharr filter using cuda")
    def scharr_filter_helper():
        nonlocal filtered_image
        dfilter[blockspergrid, threadsperblock](input_image, filtered_image, SCHARR_X, SCHARR_Y)
        cuda.synchronize()

    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        scharr_filter_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

   

    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Sharr Filter Result')
    plt.axis('off')
    plt.show()
    
    print("Running Laplace filter using cuda")
    
    
    def Laplace_filter_helper():
        nonlocal filtered_image, blurred_image
        gfilter[blockspergrid, threadsperblock](input_image, filtered_image, Gausskernel)
        opfilter[blockspergrid, threadsperblock](input_image, filtered_image, Laplacekernel, Gausskernel, blurred_image)
        cuda.synchronize()
    
    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        Laplace_filter_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 
    
    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Laplace Blur Result')
    plt.axis('off')
    plt.show()
    
    print("Running Canny filter using cuda")
    def Canny_filter_helper():
        nonlocal filtered_image, blurred_image
        gfilter[blockspergrid, threadsperblock](input_image, filtered_image, Gausskernel)
        cfilter[blockspergrid, threadsperblock](input_image, filtered_image, CANNY_X, CANNY_Y, Gausskernel, blurred_image)
        cuda.synchronize()
    
    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        Canny_filter_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 
    
    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Canny Blur Result')
    plt.axis('off')
    plt.show()
   
    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame(columns=['Filter', 'Time'])
     
    # Read the image using matplotlib
    print("Running sobel filter using cuda")
    # Load image
    image_path = 'images/im1.png'  # Update with your image path
    input_image = mpimg.imread(image_path)
     
    # Check if the image is in color (3 channels)
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        # Convert to grayscale using np.dot
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])
     
    input_image = input_image.astype(np.float32)
     
    # Define block size and grid size
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(input_image.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(input_image.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
     
    # Define Sobel kernels
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32)
    SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32)
    
    # Define Prewitt kernels
    PREWITT_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32)
    PREWITT_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32)
    
    # Define Scharr kernels
    SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.int32)
    SCHARR_Y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.int32)
    
    # Gaussian Kernel
    Gausskernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.int32) / 16
    
    # Laplace Kernel
    Laplacekernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.int32)
    
    # Canny operator kernels
    CANNY_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    CANNY_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
     
    filtered_image = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    blurred_image = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
     
   
     
    # Run the helper functions and record the timings
    filters = ['sobel', 'prewitt', 'scharr', 'laplace', 'canny']
    helpers = [sobel_filter_helper, prewitt_filter_helper, scharr_filter_helper, Laplace_filter_helper, Canny_filter_helper]
     
    for filter_name, helper in zip(filters, helpers):
        times_to_run = 1
        timing = np.empty(times_to_run, dtype=np.float32)
        for i in range(timing.size):
            cuda.synchronize()
            tic = perf_counter_ns()
            helper()
            toc = perf_counter_ns()
            timing[i] = toc-tic
        timing *= 1e-6
        print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 
     
        # Add the result to the DataFrame
        results = results.append({'Filter': filter_name, 'Time': timing.mean()}, ignore_index=True)
     
    # Output the DataFrame to a CSV file
    results.to_csv('CudaImplementation.csv', index=False)
     
    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filter Result')
    plt.axis('off')
    plt.show()
     
   

if __name__ == '__main__' :
    main()