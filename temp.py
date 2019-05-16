
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
read = pd.read_csv('heart.csv')
print(read.head())
print(read.size,read.shape)
'''df =read['age']
print(df)
sa = np.array(df)
sa.sort()
uni = np.unique(sa)
print(sa,uni,uni.size)'''

sos1=0
sos2=0
x= read['age']
y=read['trestbps']
for i  in y:
    val= y[i]
    sqr = val*val
    sos1 = sos1+sqr
    
for a in x:
    val1 = x[a]
    sqr = val1*val1
    sos2 = sos2+sqr
    
print("he sum of squares of tthe trest bps is",sos1)    
print("he sum of squares of tthe ages is",sos2)
ymean = (sos1/y.size)
xmean =(sos2/x.size)
print(ymean,xmean)   
regressor = LinearRegression()
regressor.fit(x,y)
ypred = regressor.predict(x)
    
     
plt.scatter(x,y,color ='red')
plt.plot(x,regressor.predict(x),color ='blue')
plt.show()