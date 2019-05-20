import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pt 
from sklearn.neural_network import MLPClassifier 
import os
from PIL import Image
import csv

Train=pd.read_csv("F:\programming\python\ML\\numbers_rec\Train.csv").as_matrix()
Test=pd.read_csv("F:\programming\python\ML\\numbers_rec\Test.csv").as_matrix()
X_train=Train[0:,1:]
train_label=Train[0:,0]
X_test=Test[0:,1:]
test_label=Test[0:,0]
clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(500,300,))
clf.fit(X_train,train_label)
p=clf.predict(X_test)
count=0
for i in range(len(Test)):
    count+=1 if p[i]==test_label[i] else 0
print("Accuracy =",(count/len(Test)*100))

        

        
