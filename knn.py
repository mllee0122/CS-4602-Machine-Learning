
# coding: utf-8

# * **Introduction to Machine Learning HW2**
# * Due : 20181018  
# * 107064522

# In[9]:


#Computer Assignment 3-1


# In[2]:


import numpy as np
import pandas as pd
from math import sqrt
from sklearn import datasets #Iris data

#Processing Iris Dataset
iris = datasets.load_iris()
data = pd.DataFrame(iris['data'], columns = iris['feature_names'])
label = pd.DataFrame(iris['target'], columns = ['class'])
dataset = pd.concat([data, label], axis = 1)

##Implement knn
#Calculate the Euclidean Distance
def distance(iris_row, x):
    dist = 0
    for i, j in x.items():
        dist += (iris_row[i] - j)**2
    dist = sqrt(dist)
    return dist

#knn
def knn(tr_data, k, x):
    Euclidean_dist = []
    #calculate the distance between x and each data
    for index, row in tr_data.iterrows():
        Euclidean_dist.append((index, distance(row, x)))
    #sort the distance
    Euclidean_dist = sorted(Euclidean_dist, key = lambda dist:dist[1])
    #pick the k-nearest neighbors
    knn_idx = [index[0] for index in Euclidean_dist[0:k]]
    knn = tr_data.iloc[knn_idx]
    return knn['class'].value_counts().idxmax()


# In[3]:


#input some datas for testing
#using 2-nn
x1 = pd.DataFrame.from_dict({'sepal length (cm)': [4.8], 'sepal width (cm)': [3], 'petal length (cm)':[1.4], 'petal width (cm)': [0.2] })
y1 = knn(dataset, 2, x1)

x2 = pd.DataFrame.from_dict({'sepal length (cm)': [6], 'sepal width (cm)': [2.2], 'petal length (cm)':[4.5], 'petal width (cm)': [1.5] })
y2 = knn(dataset, 2, x2)

x3 = pd.DataFrame.from_dict({'sepal length (cm)': [6], 'sepal width (cm)': [2.5], 'petal length (cm)':[5.5], 'petal width (cm)': [1.3] })
y3 = knn(dataset, 2, x3)

#print the inputs and corresponding results
#display(x1)
print("Predict class of x1:", y1)
#display(x2)
print("Predict class of x2:", y2)
#display(x3)
print("Predict class of x3:", y3)


# In[4]:


#Computer Assignment 3-2


# In[5]:


#choose 60% of dataset as training data  (tr_data)
#set the rest 40% as testing data (te_data)
tr_data = pd.concat([dataset.iloc[0:30], dataset.iloc[50:80], dataset.iloc[100:130]], axis = 0)
te_data = pd.concat([dataset.iloc[30:50], dataset.iloc[80:100], dataset.iloc[130:150]], axis = 0)

#reset the index
tr_data = tr_data.reset_index()
te_data = te_data.reset_index()

#remove the index column
tr_data = tr_data.drop("index", axis = 1)
te_data = te_data.drop("index", axis = 1)

#remove the class of testing data
te_data_woc = te_data.drop("class", axis = 1)


# In[6]:


te_data['output'] = -1

for i in range(len(te_data)):
    output = knn(tr_data, 1, te_data_woc.iloc[i])
    te_data['output'][i] = output


# In[7]:


te_data.head()


# In[8]:


true = 0
false = 0
for i in range(len(te_data)):
    if (te_data.iloc[i])['class'] == (te_data.iloc[i])['output']:
            true += 1
    else:
            false += 1
performance = 100*true/(true+false)
print("Number of correct predictions: ", true)
print("Number of incorrect predictions: ", false)
print("The performance of 1-NN classifier is", round(performance, 4), "%")