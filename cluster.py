# -*- coding: utf-8 -*-
import pandas as pd  
names =['ID','LONGITUDE','Latitude','Altitude']
read = pd.read_csv('clusterdata.csv',names=names)

print(read.head())
