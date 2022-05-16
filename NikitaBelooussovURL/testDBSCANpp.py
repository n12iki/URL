import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from DBSCANpp import DBSCANPP
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time

def test(data,epsList,minPtsList,initPoints): #this function runs the DBSCAN++ and will return the best config and the best scores.
    best=[0,0]
    bestScore=[float('-inf'),float('-inf'),float('-inf')]
    timeList=[]
    for epsV in epsList:
        for minPts in minPtsList:
            scan=DBSCANPP(epsV, minPts) #initilize the class
            start_time = time.time()
            clustered=scan.fit_predict(data,n=initPoints,pointType="knn") #test config with knn point initilizing, uniform is also possible
            timeList.append(time.time()-start_time)
            cluster_df = pd.DataFrame(clustered, columns = ["cluster", "idx"])
            cluster_df=cluster_df.sort_values("idx")
            score=0
            try:
                ars=metrics.adjusted_rand_score(cluster_df["cluster"], y)#closer to 1 is good
                fms=metrics.fowlkes_mallows_score(cluster_df["cluster"], y)#closer 1 is good
                chs=metrics.calinski_harabasz_score(np.array(data), np.array(cluster_df["cluster"]))#higher is better

                if ars>bestScore[0]:
                    score=score+1
                if fms>bestScore[1]:
                    score=score+1
                if chs>bestScore[2]:
                    score=score+1
                if score>=2:
                    best=[epsV,minPts]
                    bestScore=[ars,fms,chs]
            except:
                pass
    return [best,bestScore,sum(timeList)/float(len(timeList))]

def convertData(df): #this function converts normalizes and one hot encodes the dataset
    label_encoder = LabelEncoder()
    count=0
    temp=df
    for i in df:
        count=count+1
        if type(df[i][0])!=str: #normalize columns
            if min(np.array(df.iloc[:, i]))<0:
                column = np.array(df.iloc[:, i])+abs(np.array(df.iloc[:, i]))
                temp.iloc[:,i]=column    
            column = np.array(df.iloc[:, i])/max(df.iloc[:, i])
            temp.iloc[:,i]=column
        else: #one hot encode strings, not used with current data sets but developed in case would be necessary
            #if count==len(df.columns)-1: #dont run on last column since it is ussually class
            column=df.iloc[:,i]
            integer_encoded = label_encoder.fit_transform(column)
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            temp=temp.drop(i, axis=1)
            temp=pd.concat([temp, pd.DataFrame(onehot_encoded)], axis=1)
    columnNames=list(range(len(temp.columns)))
    temp.columns = columnNames

    return temp




### Create Blob data set
centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.2, 1, 1.1]

X, y= make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
eps = .7
minPts = 3

data = pd.DataFrame(X, columns = ["X", "Y"] )
scan=DBSCANPP(eps, minPts) #initilize the DBSCAN class
start_time = time.time()
clustered=scan.fit_predict(data,n=40,pointType="knn") #clusterize the data
print("time")
print(time.time()-start_time)
cluster,idx = list(zip(*clustered))
cluster_df = pd.DataFrame(clustered, columns = ["cluster", "idx"])
cluster_df=cluster_df.sort_values("idx")
plt.figure(figsize=(10,7))
for clust in np.unique(cluster):
    plt.scatter(X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 0], X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 1], s=10, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc ="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
print("Blobs:")
print("adjusted_rand_score:")
print(metrics.adjusted_rand_score(cluster_df["cluster"], y))
print("fowlkes_mallows_score:")
print(metrics.fowlkes_mallows_score(cluster_df["cluster"], y))
print("calinski_harabasz_score")
print(metrics.calinski_harabasz_score(np.array(data["X"]).reshape(-1,1), np.array(cluster_df["cluster"])))

###_______________________________________
###Make the moon dataset
X, y= make_moons(n_samples=500,shuffle=True, noise=.15, random_state=1)
data = pd.DataFrame(X, columns = ["X", "Y"] )
eps = .13
minPts = 3
start_time = time.time()
scan=DBSCANPP(eps, minPts)#initilize class
clustered=scan.fit_predict(data,n=100,pointType="knn") #clusterize the dataset.
print("time")
print(time.time()-start_time)
cluster,idx = list(zip(*clustered))
cluster_df = pd.DataFrame(clustered, columns = ["cluster", "idx"])
cluster_df=cluster_df.sort_values("idx")

plt.figure(figsize=(10,7))
for clust in np.unique(cluster):
    plt.scatter(X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 0], X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 1], s=10, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc ="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
print("\n\n")
print("Moons:")
print("adjusted_rand_score:")
print(metrics.adjusted_rand_score(cluster_df["cluster"], y))
print("fowlkes_mallows_score:")
print(metrics.fowlkes_mallows_score(cluster_df["cluster"], y))
print("calinski_harabasz_score")
print(metrics.calinski_harabasz_score(np.array(data["X"]).reshape(-1,1), np.array(cluster_df["cluster"])))



####################################
#the Iris dataset
###################################
dataset="iris"
print("\n\n")
print(dataset)
data=pd.read_csv(dataset+'.csv', sep=',',header=None)
y=data[4]
data=data.drop([4],axis=1)
data=convertData(data)

epsList = np.arange(.01, .10, 0.01, dtype=float) #range of core radius tested
minPtsList = [2,3,5,7,10] #list of min points
scores=["adjusted rand score","fowlkes mallows score","calinski harabasz score"]

results=test(data,epsList,minPtsList,10) #start testing the different configs with the dataset
best=results[0]
bestScore=results[1]
avgTime=results[2]
print("time: "+str(avgTime))
print(best)
scores=["adjusted rand score","fowlkes mallows score","calinski harabasz score"]
for i in range(len(bestScore)):
    print(scores[i])
    print(bestScore[i])
print("\n\n\n")


##########################
###the pima diabetes dataset
###################################
dataset="pima_diabetes"
print("\n\n")
print(dataset)
data=pd.read_csv(dataset+'.csv', sep=',',header=None)
y=data[8]
data=data.drop([8],axis=1)
data=convertData(data)
epsList = np.arange(.001, .2, 0.001, dtype=float) #core distance measured
minPtsList = [2,3,5,7] #min amount of points
results=test(data,epsList,minPtsList,10) #testing configs
best=results[0]
bestScore=results[1]
avgTime=results[2]
print("time: "+str(avgTime))
print(best)
scores=["adjusted rand score","fowlkes mallows score","calinski harabasz score"]
for i in range(len(bestScore)):
    print(scores[i])
    print(bestScore[i])
print("\n\n\n")


############################
#FIre Dataset
###########################
dataset="Fire"
print("\n\n")
print(dataset)
data=pd.read_csv(dataset+'.csv', sep=',',header=None)
y=data[6]
data=data.drop([6],axis=1)
data=convertData(data)
scores=["adjusted rand score","fowlkes mallows score","calinski harabasz score"]

epsList = np.arange(.01, .25, 0.01, dtype=float)#list of core radiuses
minPtsList = [5,7,8,9,10] #min points
results=test(data,epsList,minPtsList,500) #test configs
best=results[0]
bestScore=results[1]
avgTime=results[2]
print("time: "+str(avgTime))
print(best)
scores=["adjusted rand score","fowlkes mallows score","calinski harabasz score"]
for i in range(len(bestScore)):
    print(scores[i])
    print(bestScore[i])
print("\n\n\n")




