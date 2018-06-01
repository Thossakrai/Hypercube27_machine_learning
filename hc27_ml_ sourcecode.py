# -*- coding: utf-8 -*-

#demo 1 
#from sklearn import datasets
#import pandas as pd
#
#iris = datasets.load_iris()
#print(iris.keys()) 
#
#data = iris.data
#target = iris.target
#
#df = pd.DataFrame(data, columns = iris.feature_names)
#print(df.head())

#บรรทัดที่17 พิมพ์ลงใน console
#pd.plotting.sctter_matrix(df, c = y, figsize= [8,8], s = 150]


#demo 2

#from sklearn import datasets 
#from sklearn.neighbors import KNeighborsClassifier
#import numpy as np
#
#iris = datasets.load_iris()
#
#knn  = KNeighborsClassifier(n_neighbors = 6)
#knn.fit(iris.data, iris.target)
#
#unknown_data  = np.array([[5.1, 3.8, 0.9, 2.2]])
#prediction = knn.predict(unknown_data)
#print("predict = ", prediction)


#demo3 

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import datasets
#from sklearn.model_selection import train_test_split
#
#iris = datasets.load_iris()
#x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3)
#knn = KNeighborsClassifier(n_neighbors = 8)
#knn.fit(x_train, y_train)
#prediction = knn.predict(x_test)
#print("prediction = ", prediction)
#score = knn.score(x_test, y_test)
#print("model score = ", score)
