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
#这种方法会让两个相近的数字比两个相远的更加相似 但是事实情况并非如此
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
housing_text=housing['ocean_proximity']
housing_text_encoded=encoder.fit_transform(housing_text)
housing_text_encoded
encoder.classes_
#这里采用独热码的方式 就不会出现上一步的问题了
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
housing_text_1hot=encoder.fit_transform(housing_text_encoded.reshape(-1,1)) #返回scipy的稀疏矩阵
housing_text_1hot.toarray()
#这一布可以一步到位 完成两个转换 从文本类别到整数类别 再从整数转换到独热向量
from sklearn.preprocessing import LabelBinarizer
encoder=LabelBinarizer()
housing_text_1hot=encoder.fit_transform(housing_text)
housing_text_1hot #这里返回的是密集Numpy矩阵

##流水线和转换器
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

rooms_ix, bedrooms_ix, population_ix, household_ix = [list(housing.columns).index(col)  for col in ("total_rooms", "total_bedrooms", "population", "households")]

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False, kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)
housing_extra_attribs = pd.DataFrame(housing_extra_attribs,columns=list(housing.columns)+["rooms_per_household", "population_per_household"],index=housing.index)
housing_extra_attribs.head()

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
from sklearn.compose import ColumnTransformer
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape

#选择线性回归模型进行训练
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
some_data=housing.iloc[:5] #挑选训练集的数据试试看
some_labels=housing_labels.iloc[:5]
some_data_prepared=full_pipeline.transform(some_data)
lin_reg.predict(some_data_prepared)
some_labels #两者相比差的有点多

#测量RMSE 看看训练效果
from sklearn.metrics import mean_squared_error
housing_predictions=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions) #计算实际和预测的误差
lin_rmse=np.sqrt(lin_mse)
lin_rmse

#换一个模型 决策树模型
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
 #进行交叉验证CV 对模型进行训练和评估
 from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

def display_scores(scores):
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print("Standard deviation",scores.std())

tree_rmse_scores=np.sqrt(-scores)
display_scores(tree_rmse_scores)