import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def func(item):
    if item == '是':
        return 1
    else:
        return 0


# 似然函数
def l_beta(x_hat, y, beta):
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return np.sum(-y * np.dot(x_hat, beta) + np.log(1 + np.exp(np.dot(x_hat, beta))))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient(x_hat, y, beta):
    y = y.reshape(-1, 1)    # 行向量转为列向量
    beta = beta.reshape(-1, 1)  # 行向量转为列向量
    p1 = sigmoid(np.dot(x_hat, beta))   # 列向量
    return -np.sum(x_hat * (y - p1), axis=0)    # 行向量


# 梯度下降法
def update_parameters_gradDesc(beta, x_hat, y, learning_rate, num_iterations):
    for i in range(num_iterations):
        grad = gradient(x_hat, y, beta)
        beta -= learning_rate * grad
        if i % 50 == 0:
            print('{}th iteration, likelihood function is {}\n'.format(i, l_beta(x_hat, y, beta)))
    return beta


def hessian(x_hat, y, beta):
    y = y.reshape(-1, 1)    # 行向量转为列向量
    beta = beta.reshape(-1, 1)  # 行向量转为列向量
    p1 = sigmoid(np.dot(x_hat, beta))   # 列向量
    m = x_hat.shape[0]
    P = np.eye(m) * p1 * (1 - p1)
    return np.dot(np.dot(x_hat.T, P), x_hat)    # 矩阵


def update_parameters_newton(beta, x_hat, y, num_iterations):
    for i in range(num_iterations):
        grad = gradient(x_hat, y, beta)
        hess = hessian(x_hat, y, beta)
        beta -= np.dot(np.linalg.inv(hess), grad)
        if i % 50 == 0:
            print('{}th iteration, likelihood function is {}'.format(i, l_beta(x_hat, y, beta)))
    return beta


def init_beta(n):
    return np.random.randn(n+1)


def logistic_regression(x_hat, y, method, learning_rate, num_iterations):
    '''
    w: 权重向量（行向量）
    x: 自变量（一行为一个示例）
    b: 截距项（常数）

    return:
    beta = [w, b]
    '''
    beta = init_beta(x.shape[1])
    if method == 'gradDesc':
        return update_parameters_gradDesc(beta, x_hat, y, learning_rate, num_iterations)
    elif method == 'newton':
        return update_parameters_newton(beta, x_hat, y, num_iterations)


data = pd.read_csv('watermelon3_0_Ch.csv')
data = data.iloc[:, 7:]
data['好瓜'] = data['好瓜'].map(func)
x = data.iloc[:, :2].values
y = data.iloc[:, -1].values
x_hat = np.append(x, np.ones((x.shape[0], 1)), axis=1)
beta = logistic_regression(x_hat, y, method='newton', learning_rate=0.3, num_iterations=1000)
# 可视化模型结果
beta = beta.reshape(-1, 1)
x1 = np.arange(len(y))
y1 = sigmoid(np.dot(x_hat, beta))
lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)  # 注意sklearn的逻辑回归中，C越大表示正则化程度越低。
lr.fit(x, y)
lr_beta = np.c_[lr.coef_, lr.intercept_].T
y2 = sigmoid(np.dot(x_hat, lr_beta))

plt.plot(x1, y1, 'r-', x1, y2, 'g--', x1, y, 'b-')
plt.legend(['predict', 'sklearn_predict', 'true'])
plt.show()
