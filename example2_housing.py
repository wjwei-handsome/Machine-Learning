import os
import pandas as pd
import numpy as np 
##加载数据
housing=pd.read_csv('datasets/housing/housing.csv')
housing.info()
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()
#plt.savefig('housing_bins50.png')

##测试matplot 但是失败了 只能保存图片 不能在图形界面显示出来
'''
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot
plt.savefig('test.png')
'''

##分层抽样 分出一个测试集
housing['income_cat']=np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat'] < 5,5.0,inplace=True) #将大于5的类别合并为5
type(housing)
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=41)
split.get_n_splits()
for train_index,test_index in split.split(housing,housing['income_cat']):   
    start_train_set=housing.loc[train_index]
    start_test_set=housing.loc[test_index]

'''
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[1, 2],[3, 4], [1, 2], [3, 4],[4,5],[4,5]])
print(X)
y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2])#类别数据集10*1
print(y)

ss=StratifiedShuffleSplit(n_splits=3,test_size=0.5,random_state=42)#分成5组，测试比例为0.25

for train_index, test_index in ss.split(X, y):
    print("TRAIN_INDEX:", train_index, "TEST_INDEX:", test_index)#获得索引值
    X_train, X_test = X[train_index], X[test_index]#训练集对应的值
    y_train, y_test = y[train_index], y[test_index]#类别集对应的值
    print("X_train:",X_train)
    print("y_train:",y_train)
'''

##删除income cat属性
for set in (start_train_set,start_test_set):
    set.drop(['income_cat'],axis=1,inplace=True)
housing=start_train_set.copy()
len(start_train_set)/len(start_test_set)
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.1)
plt.savefig('diliweizhi.png')
corr_matrix=housing.corr()
corr_matrix
#将预测的属性与特征标签分离开
housing=housing.drop('median_house_value',axis=1)
housing_labels=start_train_set['median_house_value'].copy()
housing
housing_labels

#处理缺失值
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')#用中位数补全
housing_num=housing.drop('ocean_proximity',axis=1)
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X=imputer.transform(housing_num)
housing_tr=pd.DataFrame(X,columns=housing_num.columns)
housing_tr.info()
#处理文本和分类属性
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
housing_text=housing['ocean_proximity']
housing_text_encoded=encoder.fit_transform(housing_text)
housing_text_encoded
encoder.classes_
print('test')