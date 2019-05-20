import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pt 
from sklearn.neural_network import MLPClassifier 
import os
from PIL import Image
import csv

path ="F:\programming\python\ML\\numbers_rec\HW2-PersianDigits\Test\\"
cats=os.listdir(path)
ccols=["label"]
for i in range (1,401):
    ccols.append(str(i))
with open('F:\programming\python\ML\\numbers_rec\Test.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(ccols)
    for cat in cats:
        newpath=path+str(cat)
        files=os.listdir(newpath)
        for file in files:
            line=[str(cat)]
            im=Image.open(newpath+"\\"+file)  
            newim=im.resize((20, 20), Image.ANTIALIAS) 
            imarray=np.array(newim) 
            binim=np.where(imarray<255,0,1)
            for row in binim:
                for col in row:
                    line.append(col)
            filewriter.writerow(line)
