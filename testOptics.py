from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time

def test(data,minPtsList):
    bestScore=[float('-inf'),float('-inf'),float('-inf')]
    best=[0,0]
    timeList=[]
    for minPts in minPtsList:
        start_time = time.time()
        clustering = OPTICS(min_samples=minPts).fit(data)
        timeList.append(time.time()-start_time)
        cluster = clustering.labels_
        data["cluster"]=cluster
        score=0
        try:
            ars=metrics.adjusted_rand_score(data["cluster"], y)#closer to 1 is good
            fms=metrics.fowlkes_mallows_score(data["cluster"], y)#closer 1 is good
            chs=metrics.calinski_harabasz_score(np.array(data), np.array(data["cluster"]))#higher is better
            if ars>bestScore[0]:
                score=score+1
            if fms>bestScore[1]:
                score=score+1
            if chs>bestScore[2]:
                score=score+1
            if score>=2:
                best=[minPts]
                bestScore=[ars,fms,chs]
                print(data["cluster"])
                print(len(set(data["cluster"])))
        except:
            pass

    return [best,bestScore,sum(timeList)/float(len(timeList))]

def convertData(df):
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





centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.2, 1, 1.1]

X, y= make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
data = pd.DataFrame(X, columns = ["X", "Y"] )
#clustered=scan.fit_predict(data,n=40,pointType="knn")
start_time = time.time()
clustering = OPTICS(min_samples=45).fit(data)
print("time")
print(time.time()-start_time)
cluster = clustering.labels_
data["idx"]=cluster
#cluster_df = pd.DataFrame(clustering, columns = ["cluster", "idx"])
plt.figure(figsize=(10,7))
for clust in np.unique(cluster):
    plt.scatter(X[data["idx"] == clust, 0], X[data["idx"]== clust, 1], s=10, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc ="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
print("Blobs:")
print("adjusted_rand_score:")
print(metrics.adjusted_rand_score(data["idx"], y))
print("fowlkes_mallows_score:")
print(metrics.fowlkes_mallows_score(data["idx"], y))
print("calinski_harabasz_score")
print(metrics.calinski_harabasz_score(np.array(data["X"]).reshape(-1,1), np.array(data["idx"])))
print("\n\n\n")

X, y= make_moons(n_samples=500,shuffle=True, noise=.15, random_state=1)
data = pd.DataFrame(X, columns = ["X", "Y"] )
#radius of the circle defined as 0.6
start_time = time.time()
clustering = OPTICS(min_samples=25).fit(data)
print("time")
print(time.time()-start_time)
cluster = clustering.labels_
data["idx"]=cluster
#cluster_df = pd.DataFrame(clustering, columns = ["cluster", "idx"])
plt.figure(figsize=(10,7))

for clust in np.unique(cluster):
    plt.scatter(X[data["idx"] == clust, 0], X[data["idx"]== clust, 1], s=10, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc ="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
print("Moons:")
print("adjusted_rand_score:")
print(metrics.adjusted_rand_score(data["idx"], y))
print("fowlkes_mallows_score:")
print(metrics.fowlkes_mallows_score(data["idx"], y))
print("calinski_harabasz_score")
print(metrics.calinski_harabasz_score(np.array(data["X"]).reshape(-1,1), np.array(data["idx"])))
print("\n\n\n")
dataset="iris"
#radius of the circle defined as 0.6
eps = np.arange(0, .5, 0.05, dtype=float)
#minimum neighbouring points set to 3
data=pd.read_csv(dataset+'.csv', sep=',',header=None)
y=data[4]
data=data.drop([4],axis=1)
data=convertData(data)
data.columns = data.columns.astype(str)

minPtsList = np.arange(4, 20, 1)
results=test(data,minPtsList)
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





dataset="pima_diabetes"
data=pd.read_csv(dataset+'.csv', sep=',',header=None)
y=data[8]
data=data.drop([8],axis=1)
data=convertData(data)
data.columns = data.columns.astype(str)
print("pima")

minPtsList = np.arange(1, 10, 1)
best=[0,0]
scores=["adjusted rand score","fowlkes mallows score","calinski harabasz score"]
results=test(data,minPtsList)
best=results[0]
bestScore=results[1]
avgTime=results[2]
print("time: "+str(avgTime))
print(best)
for i in range(len(bestScore)):
    print(scores[i])
    print(bestScore[i])
print("\n\n\n")





dataset="Fire"
data=pd.read_csv(dataset+'.csv', sep=',',header=None)
print("Fire")
y=data[6]
data=data.drop([6],axis=1)
data=convertData(data)
data.columns = data.columns.astype(str)

minPtsList = np.arange(5, 20, 1)
best=[0,0]
scores=["adjusted rand score","fowlkes mallows score","calinski harabasz score"]
results=test(data,minPtsList)
best=results[0]
bestScore=results[1]
avgTime=results[2]
print("time: "+str(avgTime))
print(best)
for i in range(len(bestScore)):
    print(scores[i])
    print(bestScore[i])
print("\n\n\n")


