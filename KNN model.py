#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from typing import Tuple,Dict
import pandas as pd


# In[2]:


def get_distance(x:Tuple[int,int],y:Tuple[int,int])->float:
    x_diff=(y[0]-x[0])**2
    y_diff=(y[1]-x[1])**2
    return np.sqrt(x_diff+y_diff)


# In[3]:


class KNN:
    def __init__(self,dic:Dict,k):
        self.k=k
        self.df=pd.DataFrame(dic)
        
    def get_each_distance(self,test:Tuple[int,int]):
        self.df['dist']=self.df.apply(lambda row:get_distance((row.x,row.y),test),axis=1)
        
        
    def fit(self,test:Tuple[int,int]):
        self.get_each_distance(test)
        self.df=self.df.sort_values(by='dist').reset_index(drop=True)
        
        
    def get_frequent(self,label):
            d={}
            for i in label:
                if d.get(i,0):
                    d[i]+=1
                else:
                    d[i]=1
            most_frquent=None
            most_count=0
            for i,j in d.items():
                if j>most_count:
                    most_count=j
                    most_frequent=i
            return most_frequent
        
        
    def get_label(self):
        df=self.df.iloc[:self.k]
        return self.get_frequent(df['label'])
    
    
    def predict(self,test:Tuple[int,int]):
        self.fit(test)
        return self.get_label()
        

    


# In[4]:


import random

# Sample data for training
random.seed(42)  # For reproducibility
n_points = 50
data = {
    'x': [random.randint(1, 100) for _ in range(n_points)],
    'y': [random.randint(1, 100) for _ in range(n_points)],
    'label': [random.choice(['A', 'B', 'C']) for _ in range(n_points)]
}

# Create an instance of the KNN class
knn = KNN(dic=data, k=5)

# Test point outside the training data
test_point = (110, 110)

# Predict the label for the test point
predicted_label = knn.predict(test_point)
print(f'The predicted label for the test point {test_point} is {predicted_label}')

