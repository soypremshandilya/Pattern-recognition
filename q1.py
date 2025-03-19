import numpy as np

def gaussian_rbf_kernel(x, y, sigma=1.0):
    
    # Check if both vectors have the same dimension
    if len(x) != len(y):
        raise ValueError("Both vectors must have the same dimension.")
    
    # Calculate the squared Euclidean distance
    distance_squared = np.sum((x - y)**2)
    
    # Compute the Gaussian RBF Kernel
    kernel_value = np.exp(-distance_squared / (2 * sigma**2))
    
    return kernel_value



# Main Program to Take Input

if __name__ == "__main__":
    # Take input for vector x
    x = list(map(float, input("Enter the elements of vector x separated by space: ").split()))
    
    # Take input for vector y
    y = list(map(float, input("Enter the elements of vector y separated by space: ").split()))
    
    # Take input for sigma (standard deviation)
    sigma = float(input("Enter the value of sigma: "))
    
    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Compute the Gaussian RBF Kernel
    try:
        result = gaussian_rbf_kernel(x, y, sigma)
        print("\nGaussian RBF Kernel value:", result)
    except ValueError as e:
        print("\nError:", e)
