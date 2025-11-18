import pennylane as qml
import pennylane.numpy as pnp   # <-- usar o numpy do PennyLane
from sklearn.metrics import accuracy_score

class VariationalClassifier:
    def __init__(self, n_qubits=2, stepsize=0.2, epochs=80, seed=0, noise=False):
        self.n_qubits = n_qubits
        self.stepsize = stepsize
        self.epochs = epochs
        self.rng = pnp.random.default_rng(seed)
        self.noise = noise

        # Inicializa dispositivo quântico
        dev_name = "default.mixed" if noise else "default.qubit"
        self.dev = qml.device(dev_name, wires=n_qubits)

        # Parâmetros do circuito (treináveis)
        self.params = pnp.array(
            self.rng.normal(0, 0.1, size=2 * n_qubits),
            requires_grad=True
        )

        # Otimizador
        self.opt = qml.GradientDescentOptimizer(stepsize=self.stepsize)

        # Define circuito como método interno
        self.circuit = qml.QNode(self._circuit, self.dev)

    def _circuit(self, x, params):
        # Feature encoding
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)

        # Ansatz
        if self.n_qubits >= 2:
            qml.CNOT(wires=[0, 1])
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
            qml.RZ(params[i + self.n_qubits], wires=i)

        # Canais de ruído (se ativado)
        if self.noise:
            qml.AmplitudeDamping(gamma=0.05, wires=0)
            if self.n_qubits > 1:
                qml.PhaseDamping(gamma=0.03, wires=1)

        return qml.expval(qml.PauliZ(0))

    def _proba(self, X, params):
        return (pnp.array([self.circuit(x[:self.n_qubits], params) for x in X]) + 1) / 2

    def _loss(self, X, y, params):
        p = self._proba(X, params)
        eps = 1e-8
        return -pnp.mean(y * pnp.log(p + eps) + (1 - y) * pnp.log(1 - p + eps))

    def fit(self, X, y):
        params = self.params
        for _ in range(self.epochs):
            params = self.opt.step(lambda p: self._loss(X, y, p), params)
        self.params = params
        return self

    def predict(self, X):
        p = self._proba(X, self.params)
        return (p > 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
