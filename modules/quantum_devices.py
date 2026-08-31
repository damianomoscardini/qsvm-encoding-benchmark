import pennylane as qml

# ------------------------------------------------------------------------------------------------

# STANDARD LIGHTNING_QUBIT DEVICE.

def lightning_qubit_device(number_qubits):

    return qml.device("lightning.qubit", wires=number_qubits)

# ------------------------------------------------------------------------------------------------

# DEVICE THAT ALLOCATES THE QUBITS ON THE GPU IF A CERTAIN THRESHOLD IS REACHED.

def allocate_GPU_device(number_qubits):

    GPU_THRESHOLD = 16

    try:
        if number_qubits >= GPU_THRESHOLD:

            print(f"Allocated {number_qubits} qubits. Backend: GPU.")
            
            return qml.device("lightning.gpu", wires=number_qubits)
        
        else:

            print(f"Allocated {number_qubits} qubits. Backend: CPU.")

            return qml.device("lightning.qubit", wires=number_qubits)
        
    except qml.DeviceError:

        print("GPU not available. Fallback on CPU.")

        return qml.device("lightning.qubit", wires=number_qubits)