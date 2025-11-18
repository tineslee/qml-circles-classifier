from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.svm import SVC

def train_qsvm(Xtr, ytr, Xte, yte):
    # Define backend (simulador clássico)
    backend = BasicAer.get_backend("qasm_simulator")

    # Cria QuantumInstance com parâmetros fixos
    qi = QuantumInstance(
        backend=backend,
        shots=1024,
        seed_simulator=42,
        seed_transpiler=42
    )

    # Kernel quântico usando simulador
    qkernel = QuantumKernel(quantum_instance=qi)

    # Avaliar matrizes de kernel
    K_tr = qkernel.evaluate(Xtr, Xtr)
    K_te = qkernel.evaluate(Xte, Xtr)

    # Treinar SVM com kernel pré-computado
    qsvm = SVC(kernel="precomputed").fit(K_tr, ytr)
    acc = qsvm.score(K_te, yte)

    return qsvm, acc
