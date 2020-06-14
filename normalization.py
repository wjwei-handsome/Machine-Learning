from sklearn import preprocessing
import numpy as np 

a=np.array([[10,2.7,3.6],[-100,5,-2],[120,20,40]],dtype=np.float64)
print(preprocessing.scale(a))

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
X,y=datasets.make_classification()