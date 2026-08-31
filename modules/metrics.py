import numpy as np

# ------------------------------------------------------------------------------------------------

# KTA CALCULATION FUNCTION.

def calculate_kernel_target_alignment(K, y):
    
    y = np.asarray(y).flatten()
    
    # Converting the targets to {-1, +1} for the KTA calculation.

    unique_values = np.unique(y)
    if not np.array_equal(np.sort(unique_values), np.array([-1, 1])):
        if len(unique_values) != 2:
            raise ValueError("calculate_kernel_target_alignment requires a 2-class problem.")
        y = np.where(y == unique_values[0], -1, 1)
    
    # Target matrix y * y^T.

    Y_target = np.outer(y, y)
    
    # Frobenius inner product between K and Y_target.

    inner_product = np.sum(K * Y_target)
    
    # Frobenius norms.

    norm_K = np.linalg.norm(K, ord='fro')
    norm_Y = np.linalg.norm(Y_target, ord='fro')
    
    # Alignment calculation.

    alignment = inner_product / (norm_K * norm_Y)
    
    return alignment

# ------------------------------------------------------------------------------------------------

# GEOMETRIC COEFFICIENT CALCULATION FUNCTION.

def calculate_geometric_coefficient(K_classic, K_quantum, lambda_reg=1e-6):

    number_samples = K_classic.shape[0]
    
    K_quantum_regularized = K_quantum + lambda_reg * np.eye(number_samples)
    
    # Calculating the inverse of the quantum kernel.

    try:

        K_quantum_inverse = np.linalg.inv(K_quantum_regularized)

    except np.linalg.LinAlgError:
        
        K_quantum_inverse = np.linalg.pinv(K_quantum_regularized)
        
    # Calculating the square root of the classic kernel.

    eigenvalues_classic, eigenvectors_classic = np.linalg.eigh(K_classic)
    
    # Truncating any negative eigenvalues due to numerical errors.

    eigenvalues_classic = np.maximum(eigenvalues_classic, 0)
    
    K_classic_sqrt = eigenvectors_classic @ np.diag(np.sqrt(eigenvalues_classic)) @ eigenvectors_classic.T
    
    # Calculating the combined matrix: sqrt(K_c) * inv(K_q) * sqrt(K_c).

    combined_matrix = K_classic_sqrt @ K_quantum_inverse @ K_classic_sqrt
    
    # The coefficient is the square root of the spectral norm of the combined matrix 
    # (the spectral norm for symmetric matrices is the maximum absolute eigenvalue).

    eigenvalues_combined_matrix = np.linalg.eigvalsh(combined_matrix)
    spectral_norm = np.max(np.abs(eigenvalues_combined_matrix))
    
    geometric_coefficient = np.sqrt(spectral_norm)
    
    return geometric_coefficient