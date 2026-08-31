import numpy as np
import pennylane as qml
from sklearn.preprocessing import normalize

# ------------------------------------------------------------------------------------------------

# KERNEL NAME AND HYPERPARAMETERS.

kernel_name = "amplitude_kernel"

def get_kernel_hyperparameters(number_features, max_qubits):

    return {}

# ------------------------------------------------------------------------------------------------

# KERNEL CALCULATION FUNCTION.

def calculate_kernel(X_dataset_1, X_dataset_2, number_features, quantum_device):

    # QUANTUM CIRCUIT DEFINITION.

    number_qubits = int(np.ceil(np.log2(number_features)))
    actual_device = quantum_device(number_qubits)

    @qml.qnode(actual_device)
    def quantum_circuit(sample_1, sample_2): 

        qml.AmplitudeEmbedding(features=sample_1, wires=range(number_qubits), normalize=True, pad_with=0.)
        qml.adjoint(qml.AmplitudeEmbedding)(features=sample_2, wires=range(number_qubits), normalize=True, pad_with=0.) 

        return qml.expval(qml.Projector(np.zeros(number_qubits), wires = range(number_qubits)))

    # If the dataset is the same, we don't have to calculate every element of the kernel matrix.

    if X_dataset_1 is X_dataset_2 or np.array_equal(X_dataset_1, X_dataset_2):

        # ADAPTING THE DATA TO THE ENCODING.

        # To adapt the dataset for amplitude encoding, we must ensure that the vectors containing the 
        # features of the samples we pass to the encoding function have unit norm.

        X_dataset_normalized = normalize(X_dataset_1, norm="l2")
    
        # CALCULATING THE KERNEL (the Gram matrix).
        
        kernel = qml.kernels.square_kernel_matrix(X_dataset_normalized, quantum_circuit, assume_normalized_kernel=True)

        # RESULTS.

        return kernel   
    
    else:

        # ADAPTING THE DATASET TO THE ENCODING.

        X_dataset_1_normalized = normalize(X_dataset_1, norm="l2")
        X_dataset_2_normalized = normalize(X_dataset_2, norm="l2")

        # CALCULATING THE KERNEL (the Gram matrix).
        
        kernel = qml.kernels.kernel_matrix(X_dataset_1_normalized, X_dataset_2_normalized, quantum_circuit)

        # RESULTS.

        return kernel