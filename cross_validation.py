from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target


# 传统分割测试集和训练集
# 测试集训练集分层
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# y_pred=knn.predict(X_test)
print(knn.score(X_test, y_test))  # 用测试集验证分数

# 交叉验证
''' #演示说明 $代表训练集 &代表测试集
整个数据集: @@@@@@@@@
CV1:      $$$$$$$$(&&)
CV2:      $$$$$$(&&)$$
CV3:      $$$$(&&)$$$$
CV4:      $$(&&)$$$$$$
CV5:      (&&)$$$$$$$$
'''
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # 这里的分数用精确度来表示
print(scores.mean())

# 可视化展示 看看参数的合适程度 确定neighbor参数
k_range=range(1,31)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy') 
    k_scores.append(scores.mean())

plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('CV Accuracy')
plt.savefig('K_CV_Accuracy.png')
plt.close('all')