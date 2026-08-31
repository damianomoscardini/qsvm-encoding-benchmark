import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel

# ------------------------------------------------------------------------------------------------

# KERNEL NAME AND HYPERPARAMETERS.

kernel_name = "projected_quantum_kernel"

def get_kernel_hyperparameters(number_features, max_qubits):

    return {"gamma": [0.1, 1.0, 5.0], "layer_number": [1, 2]}

# ------------------------------------------------------------------------------------------------

# KERNEL CALCULATION FUNCTION.

def calculate_kernel(X_dataset_1, X_dataset_2, number_features, quantum_device, gamma, layer_number):

    # QUANTUM CIRCUIT DEFINITION.

    number_qubits = number_features
    actual_device = quantum_device(number_qubits)

    def entangling_block(wires):
        n = len(wires)
        for i in range(len(wires)):
            qml.CNOT(wires = [wires[i], wires[(i + 1) % n]])

    @qml.qnode(actual_device)
    def quantum_circuit(sample):
        for _ in range(layer_number):
            qml.AngleEmbedding(sample, wires=range(number_qubits), rotation="Y") 
            entangling_block(range(number_qubits)) 

        observables = []
        for w in range(number_qubits):
            observables.extend([qml.expval(qml.PauliX(w)), qml.expval(qml.PauliY(w)), qml.expval(qml.PauliZ(w))])
        return observables
    
    # If the dataset is the same, we don't have to calculate every element of the kernel matrix.
    
    if X_dataset_1 is X_dataset_2 or np.array_equal(X_dataset_1, X_dataset_2):

        # ADAPTING THE DATA TO THE ENCODING.

        # To adapt the dataset for angle encoding, we need to make the encoding bijective: we must map 
        # the sample feature values between 0 and pi using MinMaxScaler.

        scaler = MinMaxScaler(feature_range=(0.05*np.pi, 0.95*np.pi))

        X_dataset_scaled = scaler.fit_transform(X_dataset_1)

        # CALCULATING THE KERNEL (the Gram matrix).

        phi = np.array([quantum_circuit(x) for x in X_dataset_scaled])
        
        effective_gamma=gamma / (3 * number_features)

        kernel = rbf_kernel(phi, Y=None, gamma=effective_gamma) 

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

        phi_1 = np.array([quantum_circuit(x) for x in X_dataset_1_scaled])
        phi_2 = np.array([quantum_circuit(x) for x in X_dataset_2_clipped])

        effective_gamma = gamma / (3 * number_features)

        kernel = rbf_kernel(phi_1, phi_2, gamma=effective_gamma)

        # RESULTS.

        return kernel