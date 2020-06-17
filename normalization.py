import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

a = np.array([[10, 2.7, 3.6], [-100, 5, -2], [120, 20, 40]], dtype=np.float64)
#print(preprocessing.scale(a))
#看看归一化前后，两者训练结果精确度的差异

random_state = random.randint(1,10) #创建随机的种子
#归一化操作之后
X1, y1 = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                                    random_state=random_state, n_clusters_per_class=1, scale=100)
X1 = preprocessing.minmax_scale(X1) #归一化
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)
clf = SVC()
clf.fit(X1_train, y1_train)
print(clf.score(X1_test, y1_test))

#归一化操作之前
X2, y2 = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                                    random_state=random_state, n_clusters_per_class=1, scale=100)
#X2 = preprocessing.minmax_scale(X) commit掉归一化
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3)
clf = SVC()
clf.fit(X2_train, y2_train)
print(clf.score(X2_test, y2_test))
