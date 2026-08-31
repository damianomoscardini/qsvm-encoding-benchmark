import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------------------------------

# KERNEL NAME AND HYPERPARAMETERS.

kernel_name = "basis_kernel"

def get_kernel_hyperparameters(number_features, max_qubits):

    if number_features > max_qubits:

        raise ValueError(f"The number of features, {number_features}, exceeds the maximum limit of {max_qubits}.")
    
    max_tau = max_qubits // number_features
    
    return {"tau_bit_feature": list(range(1, max_tau + 1))}

# ------------------------------------------------------------------------------------------------

# BINARY CONVERSION FUNCTION (made by Gemini).

def binary_conversion(X_dataset, tau):
    
    max_value = (2 ** tau) - 1
    integer_data = np.round(X_dataset * max_value).astype(np.uint8)
    matrix_8bit = np.unpackbits(integer_data[:, :, np.newaxis], axis=2)

    return matrix_8bit[:, :, -tau:].reshape(X_dataset.shape[0], X_dataset.shape[1] * tau)

# ------------------------------------------------------------------------------------------------

# KERNEL CALCULATION FUNCTION.

def calculate_kernel(X_dataset_1, X_dataset_2, number_features, quantum_device, tau_bit_feature):

    # QUANTUM CIRCUIT DEFINITION.

    number_qubits = number_features * tau_bit_feature
    actual_device = quantum_device(number_qubits)

    @qml.qnode(actual_device)
    def quantum_circuit(sample_1, sample_2): 

        qml.BasisEmbedding(features=sample_1, wires=range(number_qubits)) 
        qml.adjoint(qml.BasisEmbedding)(features=sample_2, wires=range(number_qubits)) 

        return qml.expval(qml.Projector(np.zeros(number_qubits), wires=range(number_qubits)))

    # If the dataset is the same, we don't have to calculate every element of the kernel matrix.

    if X_dataset_1 is X_dataset_2 or np.array_equal(X_dataset_1, X_dataset_2):

        # ADAPTING THE DATA TO THE ENCODING.

        # To adapt the dataset for angle encoding,  we must map the sample feature values between 0 and 1 using MinMaxScaler.
        # This is needed by the binary_conversion function, that only accepts values between 0 and 1.
    
        scaler = MinMaxScaler(feature_range=(0.05, 0.95))

        X_dataset_scaled = scaler.fit_transform(X_dataset_1)

        X_dataset_binary = binary_conversion(X_dataset_scaled, tau_bit_feature)

        # CALCULATING THE KERNEL (the Gram matrix).
    
        kernel = qml.kernels.square_kernel_matrix(X_dataset_binary, quantum_circuit, assume_normalized_kernel=True)

        # RESULTS.

        return kernel
    
    else:

        # ADAPTING THE DATA TO THE ENCODING.

        # Now that we have two datasets, we compute the rescaling parameters (fit) exclusively on dataset_1 to prevent data leakage, 
        # and only apply the transformation (transform) on dataset_2. However, since dataset_2 might have more extreme original 
        # values than dataset_1 and end up "out of bounds", we apply clipping using np.clip.

        scaler = MinMaxScaler(feature_range=(0.05, 0.95))

        X_dataset_1_scaled = scaler.fit_transform(X_dataset_1)
        X_dataset_2_scaled = scaler.transform(X_dataset_2)

        X_dataset_2_clipped = np.clip(X_dataset_2_scaled, 0.0, 1.0)

        fraction_clipped = np.sum(X_dataset_2_scaled != X_dataset_2_clipped) / X_dataset_2_scaled.size

        if fraction_clipped > 0.05:
            print(f"WARNING: {fraction_clipped*100:.2f}% of the validation/test data was clipped in {kernel_name}.")
        
        X_dataset_1_binary = binary_conversion(X_dataset_1_scaled, tau_bit_feature)
        X_dataset_2_binary = binary_conversion(X_dataset_2_clipped, tau_bit_feature)

        # CALCULATING THE KERNEL (the Gram matrix).
    
        kernel = qml.kernels.kernel_matrix(X_dataset_1_binary, X_dataset_2_binary, quantum_circuit)

        # RESULTS.
    
        return kernel