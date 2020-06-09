from sklearn import datasets
from sklearn.linear_model import LinearRegression
loaded_data=datasets.load_boston()
#print(loaded_data.target)
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)

print(model.predict(data_X[:4,:]))
print(data_y[:4])

##创造自己的数据
'''
wwj_X,wwj_y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
import matplotlib.pyplot as plt 
plt.scatter(wwj_X,wwj_y)
plt.savefig('wwj_X_y.png')
'''