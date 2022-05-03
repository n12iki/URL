import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs

#https://becominghuman.ai/dbscan-clustering-algorithm-implementation-from-scratch-python-9950af5eed97
def check_core(eps,minPoints,df,index):
    temp=df[1-euclidean_distances(df, df.to_numpy()[index,None])<=eps]
    if len(temp)>=minPoints:
        return (temp.index,"core")
    elif len(temp)<minPoints and len(temp)>0:
        return (temp.index,"border")
    else:
        return (temp.index,"noise")

def DBSCAN(eps,minPoints,df,points):
    C=1
    PoI=[] #points that are neighbors to core points
    visited=[]
    remaining=points
    clusters={}
    madeClust=False
    while len(remaining)!=0:
        first=True
        x=random.choice(remaining)
        PoI.add(x)
        visited.append(x)
        while len(PoI)!=0:
            index=random.choice(PoI)
            visited.append(index)
            PoI.remove(index)
            neighbors,cat=check_core(eps,minPoints,df,index)
            if cat=="core":
                madeClust=True 
                try:
                   clusters[C]=clusters[C].append(index)
                except:
                    clusters[C]=[index]
                PoI.extend(neighbors)
            elif cat=="border":
                try:
                   clusters[C]=clusters[C].append(index)
                except:
                    clusters[C]=[index]
            else:
                try:
                   clusters[0]=clusters[0].append(index)
                except:
                    clusters[0]=[index]
        
        if madeClust==True:
            C=C+1
            madeClust=False
    return clusters


centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.2, 1, 1.1]

X, y= make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
#radius of the circle defined as 0.6
eps = 0.6
#minimum neighbouring points set to 3
minPts = 3

data = pd.DataFrame(X, columns = ["X", "Y"] )
points=list(data.index)
clustered = DBSCAN(eps, minPts, data, points)

idx , cluster = list(zip(*clustered))
cluster_df = pd.DataFrame(clustered, columns = ["idx", "cluster"])

plt.figure(figsize=(10,7))
for clust in np.unique(cluster):
    plt.scatter(X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 0], X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 1], s=10, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc ="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
