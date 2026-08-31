import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# ------------------------------------------------------------------------------------------------

# KERNEL NAME AND HYPERPARAMETERS.

kernel_name = "classic_RBF_kernel"

def get_kernel_hyperparameters(number_features, max_qubits):

    return {"gamma": [0.01, 0.1, 1.0, 5.0, 10.0]}

# ------------------------------------------------------------------------------------------------

# KERNEL CALCULATION FUNCTION.

def calculate_kernel(X_dataset_1, X_dataset_2, number_features, quantum_device, gamma):

    effective_gamma = gamma / number_features

    if X_dataset_1 is X_dataset_2 or np.array_equal(X_dataset_1, X_dataset_2):

        # ADAPTING THE DATA TO THE ENCODING.

        # We don't have to adapt the data to the encoding, since the only thing that the 
        # classic RBF kernel needs is a scaled and PCA-applied dataset.

        # CALCULATING THE KERNEL (the Gram matrix).

        kernel = rbf_kernel(X_dataset_1, Y = None, gamma = effective_gamma)

        return kernel
    
    else:

        # CALCULATING THE KERNEL (the Gram matrix).

        kernel = rbf_kernel(X_dataset_1, X_dataset_2, gamma = effective_gamma)

        # RESULTS.

        return kernel