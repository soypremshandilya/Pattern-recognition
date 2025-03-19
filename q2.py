import numpy as np

def rbf_kernel(X, gamma):
   
    # Compute the squared Euclidean distance matrix
    sq_dist = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    
    # Compute the RBF kernel matrix
    K = np.exp(-gamma * sq_dist)
    
    return K

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])  # Example dataset with shape (3, 2)
gamma = 0.5

K = rbf_kernel(X, gamma)
print("RBF Kernel Matrix:")
print(K)
