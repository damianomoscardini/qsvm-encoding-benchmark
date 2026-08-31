import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------------------------------

# KERNEL NAME AND HYPERPARAMETERS.

kernel_name = "angle_kernel"

def get_kernel_hyperparameters(number_features, max_qubits):

    return {}

# ------------------------------------------------------------------------------------------------

# KERNEL CALCULATION FUNCTION.

def calculate_kernel(X_dataset_1, X_dataset_2, number_features, quantum_device):

    # QUANTUM CIRCUIT DEFINITION.
    
    number_qubits = number_features
    actual_device = quantum_device(number_qubits)

    @qml.qnode(actual_device)
    def quantum_circuit(sample_1, sample_2): 

        qml.AngleEmbedding(sample_1, wires=range(number_qubits), rotation="Y") 
        qml.adjoint(qml.AngleEmbedding)(sample_2, wires=range(number_qubits), rotation="Y") 
        
        return qml.expval(qml.Projector(np.zeros(number_qubits), wires=range(number_qubits)))

    # If the dataset is the same, we don't have to calculate every element of the kernel matrix.

    if X_dataset_1 is X_dataset_2 or np.array_equal(X_dataset_1, X_dataset_2):

        # ADAPTING THE DATA TO THE ENCODING.

        # To adapt the dataset for angle encoding, we need to make the encoding bijective: we must map 
        # the sample feature values between 0 and pi using MinMaxScaler.

        scaler = MinMaxScaler(feature_range=(0.05*np.pi, 0.95*np.pi))

        X_dataset_scaled = scaler.fit_transform(X_dataset_1)

        # CALCULATING THE KERNEL (the Gram matrix).
    
        kernel = qml.kernels.square_kernel_matrix(X_dataset_scaled, quantum_circuit, assume_normalized_kernel=True)

        # RESULTS.

        return kernel      
    
    else: 

        # ADAPTING THE DATA TO THE ENCODING.

        # Now that we have two datasets, we compute the rescaling parameters (fit) exclusively on dataset_1 to prevent data leakage, 
        # and only apply the transformation (transform) on dataset_2. However, since dataset_2 might have more extreme original 
        # values than dataset_1 and end up "out of bounds", we apply clipping using np.clip.

        scaler = MinMaxScaler(feature_range=(0.05*np.pi, 0.95*np.pi))

        X_dataset_1_scaled = scaler.fit_transform(X_dataset_1)
        X_dataset_2_scaled = scaler.transform(X_dataset_2)

        X_dataset_2_clipped = np.clip(X_dataset_2_scaled, 0.0, np.pi)

        fraction_clipped = np.sum(X_dataset_2_scaled != X_dataset_2_clipped) / X_dataset_2_scaled.size

        if fraction_clipped > 0.05:
            print(f"WARNING: {fraction_clipped*100:.2f}% of the validation/test data was clipped in {kernel_name}.")

        # CALCULATING THE KERNEL (the Gram matrix).
    
        kernel = qml.kernels.kernel_matrix(X_dataset_1_scaled, X_dataset_2_clipped, quantum_circuit)

        # RESULTS.

        return kernel