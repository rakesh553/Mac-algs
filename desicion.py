# -*- coding: utf-8 -*-
from sklearn import tree

X= [[0,0],[1,1]]
Y =[0,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

print(clf.predict([[2.,2.]]))
#array([1])

"""
Created on Wed Feb 20 20:55:43 2019

@author: Sunny
"""

