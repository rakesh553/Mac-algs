# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
from sklearn.cluster import KMeans

x=[1, 5, 1.5,8,1,9]
y= [2,8,1.8,8,0.6,11]

plt.scatter(x,y)
plt.show()

X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
print(labels)

colors = ["g.","r."]

for i in range(len(X)):
    print("coordinate:",X[i],"label",labels[i])
    plt.plot(X[i][0] , X[i][1],colors[labels[i]], markersize =10)
    
plt.scatter(centroids[:,0],centroids[:,1],marker ="x",linewidth =5)
    
plt.show()
"""
Created on Tue Feb 19 18:26:07 2019

@author: Sunny
"""
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
'''names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv('iris.csv',names=names)  

print(dataset.head(),dataset.shape)  

''''''X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 4].values  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=2)  
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)
'''