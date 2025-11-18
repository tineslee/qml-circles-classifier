# src/baselines.py
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_logistic(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    return clf, acc

def train_svm_rbf(Xtr, ytr, Xte, yte):
    clf = SVC(kernel='rbf', gamma='scale', C=1.0)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    return clf, acc
