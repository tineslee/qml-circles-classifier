# src/plots.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def decision_boundary_2d(model, X, y, title="Decision boundary"):
    h = 0.02
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    cm = ListedColormap(['#FFBBBB','#BBBBFF'])
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.6)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=ListedColormap(['#FF0000','#0000FF']), s=25, edgecolor='k')
    plt.title(title)
    plt.tight_layout()
    plt.show()
