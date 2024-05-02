import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import perf_counter_ns
import numba
import pandas as pd

# The input image should be grayscale
@numba.jit(nopython=True)
def sobel_filter(input_image):
    """
    Apply a Sobel filter to the image.
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Initialize the gradient images
    grad_x = np.zeros_like(input_image)
    grad_y = np.zeros_like(input_image)

    # Apply the filter
    for i in range(1, input_image.shape[0] - 1):
        for j in range(1, input_image.shape[1] - 1):
            grad_x[i, j] = np.sum(Gx * input_image[i-1:i+2, j-1:j+2])
            grad_y[i, j] = np.sum(Gy * input_image[i-1:i+2, j-1:j+2])

    # Normalize or clip the magnitude
    # √ (grad_x^2 + grad_y^2)        
    magnitude = np.hypot(grad_x,grad_y)
    magnitude = np.clip(magnitude, 0, 255)  # Clip to the range 0-255
    return magnitude.astype(np.uint8)  # Convert to uint8 if necessary


def prewitt_operator(input_image):
    """
    Apply a Prewitt filter to the image.
    """
    Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Initialize the gradient images
    grad_x = np.zeros_like(input_image)
    grad_y = np.zeros_like(input_image)

    # Apply the filter
    for i in range(1, input_image.shape[0] - 1):
        for j in range(1, input_image.shape[1] - 1):
            grad_x[i, j] = np.sum(Gx * input_image[i-1:i+2, j-1:j+2])
            grad_y[i, j] = np.sum(Gy * input_image[i-1:i+2, j-1:j+2])

    # Normalize or clip the magnitude
    # √ (grad_x^2 + grad_y^2)        
    magnitude = np.hypot(grad_x,grad_y)
    magnitude = np.clip(magnitude, 0, 255)  # Clip to the range 0-255
    return magnitude.astype(np.uint8)  # Convert to uint8 if necessary


def gaussian_blur(input_image):
    # 3x3 Gaussian kernel
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    # Initialize the output image
    output_image = np.zeros_like(input_image)

    # Apply the filter
    for i in range(1, input_image.shape[0] - 1):
        for j in range(1, input_image.shape[1] - 1):
            output_image[i, j] = np.sum(kernel * input_image[i-1:i+2, j-1:j+2])

    # Clip the output image to the range 0-255
    output_image = np.clip(output_image, 0, 255)

    return output_image.astype(np.uint8)

def laplace_operator(input_image):
    """
    Apply a Laplace filter to the image.
    """
    # 3x3 Laplacian kernel
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # Apply Gaussian blur to the image
    blurred_image = gaussian_blur(input_image)

    # Initialize the output image
    output_image = np.zeros_like(blurred_image)

    # Apply the filter
    for i in range(1, blurred_image.shape[0] - 1):
        for j in range(1, blurred_image.shape[1] - 1):
            output_image[i, j] = np.sum(kernel * blurred_image[i-1:i+2, j-1:j+2])

    # Normalize or clip the magnitude
    magnitude = np.abs(output_image)
    magnitude = np.clip(magnitude, 0, 255)  # Clip to the range 0-255

    return magnitude.astype(np.uint8)

def canny_operator(input_image):
    """
    Apply a Canny filter to the image.
    """
    # Apply Gaussian blur to the image
    blurred_image = gaussian_blur(input_image)

    # Initialize the gradient images
    grad_x = np.zeros_like(blurred_image)
    grad_y = np.zeros_like(blurred_image)

    # canny operator kernels
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply the Sobel operator to get the gradient intensity representations of the image
    for i in range(1, blurred_image.shape[0] - 1):
        for j in range(1, blurred_image.shape[1] - 1):
            grad_x[i, j] = np.sum(Gx * blurred_image[i-1:i+2, j-1:j+2])
            grad_y[i, j] = np.sum(Gy * blurred_image[i-1:i+2, j-1:j+2])

    # Calculate the magnitude of the gradient
    magnitude = np.hypot(grad_x, grad_y)

    # Normalize or clip the magnitude
    magnitude = np.clip(magnitude, 0, 255)  # Clip to the range 0-255

    return magnitude.astype(np.uint8)

def sharr_operator(input_image):
    """
    Apply a Scharr filter to the image.
    """
    Gx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    Gy = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])

    # Initialize the gradient images
    grad_x = np.zeros_like(input_image)
    grad_y = np.zeros_like(input_image)

    # Apply the filter
    for i in range(1, input_image.shape[0] - 1):
        for j in range(1, input_image.shape[1] - 1):
            grad_x[i, j] = np.sum(Gx * input_image[i-1:i+2, j-1:j+2])
            grad_y[i, j] = np.sum(Gy * input_image[i-1:i+2, j-1:j+2])

    # Normalize or clip the magnitude
    # √ (grad_x^2 + grad_y^2)        
    magnitude = np.hypot(grad_x,grad_y)
    magnitude = np.clip(magnitude, 0, 255)  # Clip to the range 0-255
    return magnitude.astype(np.uint8)  # Convert to uint8 if necessary

def main():
    # Read the image using matplotlib
    print("Running sobel filter")
    # Load image
    image_path = 'images/im1.png'  # Update with your image path
    input_image = mpimg.imread(image_path)

    # Convert to grayscale if the image is in color
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])

    input_image = input_image.astype(np.float32) #
    filtered_image = None
    
    def sobel_filter_helper():
        nonlocal filtered_image
        filtered_image = sobel_filter(input_image)
   

    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
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
    
    
    print("Running prewitt filter")

    def prewitt_operator_helper():
        nonlocal filtered_image
        filtered_image = prewitt_operator(input_image)
        
    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        prewitt_operator_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('sharr operator Filter Result')
    plt.axis('off')
    plt.show()        
        
    print("Running sharr filter")

    def sharr_operator_helper():
        nonlocal filtered_image
        filtered_image = sharr_operator(input_image)
        
    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        prewitt_operator_helper()
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
        
        
        
    print("Running laplace filter")
    def laplace_operator_helper():
        nonlocal filtered_image
        filtered_image = laplace_operator(input_image)
    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        laplace_operator_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('laplace Result')
    plt.axis('off')
    plt.show()        
        
        
    print("Running canny filter")
    
    def canny_operator_helper():
        nonlocal filtered_image
        filtered_image = canny_operator(input_image)
    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        canny_operator_helper()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    # Display the result
    plt.rcParams["figure.figsize"] = (8,8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('canny Operator Result')
    plt.axis('off')
    plt.show()        



    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame(columns=['Filter', 'Time'])

    # Read the image using matplotlib
    print("Running sobel filter")
    # Load image
    image_path = 'images/im1.png'  # Update with your image path
    input_image = mpimg.imread(image_path)

    # Convert to grayscale if the image is in color
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])

    input_image = input_image.astype(np.float32) #
    filtered_image = None
    
    def sobel_filter_helper():
        nonlocal filtered_image
        filtered_image = sobel_filter(input_image)
   
    # Run the helper functions and record the timings
    filters = ['sobel', 'prewitt', 'sharr', 'laplace', 'canny']
    helpers = [sobel_filter_helper, prewitt_operator_helper, sharr_operator_helper, laplace_operator_helper, canny_operator_helper]

    for filter_name, helper in zip(filters, helpers):
        times_to_run = 1
        timing = np.empty(times_to_run, dtype=np.float32)
        for i in range(timing.size):
            tic = perf_counter_ns()
            helper()
            toc = perf_counter_ns()
            timing[i] = toc-tic
        timing *= 1e-6
        print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

        # Add the result to the DataFrame
        results = results.append({'Filter': filter_name, 'Time': timing.mean()}, ignore_index=True)

    # Output the DataFrame to a CSV file
    results.to_csv('baseline metrics.csv', index=False)

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