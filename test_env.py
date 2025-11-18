import pennylane as qml
import numpy as np
from qiskit.utils import algorithm_globals

# Teste PennyLane
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))

print("PennyLane test:", circuit(np.pi/4))

# Teste Qiskit
print("Qiskit algorithm_globals random int:", algorithm_globals.random.integers(0, 10))


