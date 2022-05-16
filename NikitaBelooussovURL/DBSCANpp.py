import numpy as np
import random
import math
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import time

class DBSCANPP:
    #Parameters

    #eps: min radius for finding minPts
    #minPts: minpoints in radius to be for a point to be core

    def __init__(self, eps, minPts):
        self.eps=eps
        self.minPts = minPts
    

    def fit_predict(self, data, pointType="uniform",n=10): #determins points to check for core and starts the DBSCAN
        x, y = data.shape
        if pointType=="uniform":
            step = int(x/y)
            points=list(np.array(data.index)[0::step])
        if pointType=="knn":
            points=self.k_center(n,data,x,y)
        clustered = self.DBSCAN(data, points)
        return clustered

    def check_core(self,df,index): #checks if point is a core point
        eps=self.eps
        minPoints=self.minPts

        temp=df[abs(euclidean_distances(df, df.to_numpy()[index,None]))<=eps]
        if len(temp)>=minPoints+1:
            return (temp.index,"core")
        elif len(temp)<minPoints+1 and len(temp)>0:
            return (temp.index,"border")
        else:
            return (temp.index,"noise")

    def DBSCAN(self,df,points): #the DBSCAN part of the code
        C=1
        toDo=set()
        unclaimed=list(df.index)
        clusters=[]
        while len(points)!=0:
            clusterCreated=False
            toDo.add(random.choice(points))
            while len(toDo)!=0:
                idx=toDo.pop()
                try:
                    points.remove(idx)
                except:
                    pass
                neigh,typ=self.check_core(df,idx)


                if typ=="core":
                    clusterCreated=True
                    clusters.append((C,idx))
                    unclaimed.remove(idx)
                    neigh=neigh.intersection(set(unclaimed))
                    toDo.update(neigh)


                elif typ=="border" and clusterCreated==True:
                    clusters.append((C,idx))
                    unclaimed.remove(idx)
                    neigh=neigh.intersection(set(unclaimed))
                    toDo.update(neigh)

            if clusterCreated==True:
                C=C+1
        for i in unclaimed:
            clusters.append((0,i))
        return clusters
    
    def k_center(self,n,df,x): #finds the points that should be checked as cores if the setting is knn
        indices=[]
        distances=np.empty(x)
        for i,j in df.iterrows():
            distances[i]=np.sum(euclidean_distances(df, df.to_numpy()[i,None]))
        indices.append(np.argmin(distances))
        for k in range(n-1):
            distances=np.zeros(n)
            for i in range(len(indices)):
                distances[i]=np.sum(euclidean_distances(df, df.to_numpy()[indices[i],None]))+distances[i]
            index=np.argmax(distances)            
            adjust=np.count_nonzero(indices<=index)
            testInd=[]
            while index+adjust in indices:
                testInd.append(index)
                distances=np.delete(distances,index)
                index=np.argmax(distances)
                adjust=np.count_nonzero(indices<=index)+np.count_nonzero(testInd<=index)

            indices.append(index+adjust)
        return indices


        