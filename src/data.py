# src/data.py
from sklearn.datasets import make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_circles(n_samples=400, noise=0.1, factor=0.5, test_size=0.3, seed=42):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=seed)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return Xtr, Xte, ytr, yte

def load_breast(test_size=0.3, seed=42):
    data = load_breast_cancer()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return Xtr, Xte, ytr, yte
