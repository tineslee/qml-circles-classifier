# Quantum Machine Learning: Classificação de Círculos com PennyLane e Qiskit
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

Este projeto implementa e compara diferentes modelos de classificação — clássicos e quânticos — aplicados ao problema de separação de círculos concêntricos. É uma introdução prática ao uso de computação quântica em Machine Learning.

## Modelos incluídos

- Regressão Logística (clássico)
- SVM com kernel RBF (clássico)
- VQC com PennyLane (com e sem ruído)
- QSVM com Qiskit

---

## Algoritmos utilizados

### 🔹 ![Logistic Regression](https://img.shields.io/badge/Algoritmo-Logistic%20Regression-blue)
- **Regressão Logística (Clássico):** modelo estatístico simples para classificação binária.
- Usa a função sigmoide para modelar probabilidades.
- Serve como baseline para comparação com modelos mais complexos.

---

### 🔹 ![SVM RBF](https://img.shields.io/badge/Algoritmo-SVM%20RBF-orange)
- **Support Vector Machine com Kernel RBF (Clássico):**
- Encontra o hiperplano ótimo para separar classes.
- Kernel RBF permite lidar com dados não linearmente separáveis (como círculos concêntricos).

---

### 🔹 ![VQC](https://img.shields.io/badge/Algoritmo-VQC%20(PennyLane)-purple)
- **Variational Quantum Classifier (Quântico):**
- Circuito quântico variacional com parâmetros treináveis.
- Otimizado por gradiente para aprender fronteiras de decisão.
- Testado **com e sem ruído** para simular dispositivos ideais e reais.

---

### 🔹 ![QSVM](https://img.shields.io/badge/Algoritmo-QSVM%20(Qiskit)-green)
- **Quantum Support Vector Machine (Quântico):**
- Usa **quantum kernels** para calcular similaridade entre vetores em espaço de Hilbert quântico.
- Explora representações de dados inacessíveis para kernels clássicos.
- Implementado com Qiskit, simulando execução em hardware quântico.

---
  
## Resultados

Cada modelo gera um gráfico com a fronteira de decisão sobre os dados. A acurácia é calculada e exibida para comparação. O VQC sem ruído apresenta ótimo desempenho, enquanto o VQC com ruído mostra os efeitos da decoerência.

---

## Comparação de Resultados

| Algoritmo | Acurácia | Observações |
|-----------|----------|-------------|
| ![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-85%25-blue) | 85% | Modelo clássico simples, usado como baseline. |
| ![SVM RBF](https://img.shields.io/badge/SVM%20RBF-92%25-orange) | 92% | Excelente para dados não linearmente separáveis. |
| ![VQC PennyLane](https://img.shields.io/badge/VQC%20(PennyLane)-90%25-purple) | 90% | Circuito quântico variacional sem ruído, bom desempenho. |
| ![VQC PennyLane Noise](https://img.shields.io/badge/VQC%20(PennyLane%20com%20ruído)-78%25-lightgrey) | 78% | Mostra os efeitos da decoerência e erros quânticos. |
| ![QSVM Qiskit](https://img.shields.io/badge/QSVM%20(Qiskit)-88%25-green) | 88% | Usa quantum kernels, desempenho competitivo. |

---

## Requisitos

Instale os pacotes necessários com:

```bash
pip install -r requirements.txt
```
## Execução
Para rodar todos os experimentos e visualizar os gráficos:
```bash
python3 -m experiments.run_all
```

---

## Estrutura do projeto

qml-project/

├── experiments/
│   └── run_all.py

├── src/
│   ├── qml_pennylane.py
│   └── qml_qiskit.py

├── requirements.txt

├── test_env.py

└── .gitignore

---

## Conclusões

Este projeto mostrou na prática como algoritmos **clássicos** e **quânticos** podem ser aplicados a um problema simples de classificação não-linear (círculos concêntricos).  

- **Modelos clássicos** (Regressão Logística e SVM RBF) continuam sendo extremamente eficientes e alcançaram alta acurácia, especialmente o SVM com kernel RBF.  
- **Modelos quânticos** (VQC e QSVM) já demonstram desempenho competitivo, mesmo em simulações, e trazem novas formas de representar dados.  
- O **VQC com ruído** evidenciou os desafios atuais da computação quântica, como decoerência e erros, mas também reforçou a importância de estudar estratégias de mitigação.  
- O **QSVM** mostrou que kernels quânticos podem oferecer vantagens em certos cenários, abrindo caminho para aplicações futuras em dados complexos.  

Em resumo: os algoritmos clássicos ainda lideram em confiabilidade, mas os quânticos já são promissores e representam uma **fronteira de pesquisa** que pode transformar áreas como saúde, finanças e ciência de dados nos próximos anos.


## Sobre

Projeto desenvolvido por Thais Ines.
