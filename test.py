import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from DBSCANpp import DBSCANPP
import numpy as np
from sklearn import metrics

centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.2, 1, 1.1]

X, y= make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
#radius of the circle defined as 0.6
eps = .7
#minimum neighbouring points set to 3
minPts = 3

import pandas as pd
data = pd.DataFrame(X, columns = ["X", "Y"] )
scan=DBSCANPP(.5, eps, minPts)
points=DBSCANPP.k_center(5,data)
print(data["X"][points].values)
print(data["Y"][points].values)
plt.scatter(data["X"].values, data["Y"].values, s=10, label=f"sel")
plt.scatter(data["X"][points].values, data["Y"][points].values, s=10, label=f"sel")
#plt.scatter(X[points]], y[points], s=10, label=f"sel")
plt.legend(loc ="upper right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()