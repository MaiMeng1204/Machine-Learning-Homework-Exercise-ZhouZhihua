import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut, KFold


data = np.loadtxt('Transfusion.txt', skiprows=3, delimiter=',').astype(int)
X = data[:, :4]
y = data[:, 4]
n = X.shape[0]

# normalization
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 10折交叉验证法
kf = KFold(n_splits=10, shuffle=True)
accuracy = 0
for train_index, test_index in kf.split(X):
    lr = linear_model.LogisticRegression(C=2)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr.fit(X_train, y_train)
    accuracy += lr.score(X_test, y_test)
print('10折交叉验证法准确率：{:.2%}'.format(accuracy / 10))

# 留一法
loo = LeaveOneOut()

accuracy = 0
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    lr = linear_model.LogisticRegression(C=2)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr.fit(X_train, y_train)
    accuracy += lr.score(X_test, y_test)

print('留一法准确率：{:.2%}'.format(accuracy / n))
