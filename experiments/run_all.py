from src.data import load_circles
from src.baselines import train_logistic, train_svm_rbf
from src.qml_pennylane import VariationalClassifier
from src.qml_qiskit import train_qsvm
from src.plots import decision_boundary_2d

# 1. Carregar dados
Xtr, Xte, ytr, yte = load_circles()

# 2. Treinar modelos clássicos
_, log_acc = train_logistic(Xtr, ytr, Xte, yte)
_, svm_acc = train_svm_rbf(Xtr, ytr, Xte, yte)

# 3. Treinar modelo quântico PennyLane (sem ruído)
vqc_clean = VariationalClassifier(n_qubits=2, epochs=80, noise=False).fit(Xtr, ytr)
vqc_clean_acc = vqc_clean.score(Xte, yte)

# 4. Treinar modelo quântico PennyLane (com ruído)
vqc_noisy = VariationalClassifier(n_qubits=2, epochs=80, noise=True).fit(Xtr, ytr)
vqc_noisy_acc = vqc_noisy.score(Xte, yte)

# 5. Treinar QSVM com Qiskit
_, qsvm_acc = train_qsvm(Xtr, ytr, Xte, yte)

# 6. Mostrar resultados
print("🔍 Comparação de acurácia:")
print(f"Logistic Regression:     {log_acc:.3f}")
print(f"SVM RBF:                 {svm_acc:.3f}")
print(f"PennyLane VQC (limpo):   {vqc_clean_acc:.3f}")
print(f"PennyLane VQC (ruído):   {vqc_noisy_acc:.3f}")
print(f"Qiskit QSVM:             {qsvm_acc:.3f}")

# 7. Visualizar fronteira de decisão (apenas para dados 2D)
decision_boundary_2d(vqc_clean, Xtr, ytr, title="PennyLane VQC (sem ruído)")
decision_boundary_2d(vqc_noisy, Xtr, ytr, title="PennyLane VQC (com ruído)")
