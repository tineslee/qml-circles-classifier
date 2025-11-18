# Quantum Machine Learning: Classificação de Círculos com PennyLane e Qiskit
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Copilot](https://img.shields.io/badge/feito%20com-Copilot💙-purple)

Este projeto implementa e compara diferentes modelos de classificação — clássicos e quânticos — aplicados ao problema de separação de círculos concêntricos. É uma introdução prática ao uso de computação quântica em Machine Learning.

## Modelos incluídos

- Regressão Logística (clássico)
- SVM com kernel RBF (clássico)
- VQC com PennyLane (com e sem ruído)
- QSVM com Qiskit

## Resultados

Cada modelo gera um gráfico com a fronteira de decisão sobre os dados. A acurácia é calculada e exibida para comparação. O VQC sem ruído apresenta ótimo desempenho, enquanto o VQC com ruído mostra os efeitos da decoerência.

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

## Sobre

Projeto desenvolvido por Thais Ines. Explorando o potencial da computação quântica aplicada à inteligência artificial.
