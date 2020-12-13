# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA():
    def fit(self, X, y):
        pos = y == 1
        neg = y == 0
        X0 = X[neg]
        X1 = X[pos]

        u0 = np.mean(X0, axis=0, keepdims=True)     # (1, n)
        u1 = np.mean(X1, axis=0, keepdims=True)

        Sw = np.dot((X0 - u0).T, X0 - u0) + np.dot((X1 - u1).T, X1 - u1)
        w = np.dot(np.linalg.inv(Sw), (u0 - u1).T).reshape(1, -1)   # (1, n)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))

        plt.scatter(X1[:, 0], X1[:, 1], c='r', marker='o', label='good')
        plt.scatter(X0[:, 0], X0[:, 1], c='k', marker='x', label='bad')

        plt.xlabel('密度', labelpad=1)
        plt.ylabel('含糖量')

        x_tmp = np.linspace(-0.05, 0.15)
        y_tmp = x_tmp * w[0, 1] / w[0, 0]
        plt.plot(x_tmp, y_tmp, '#808080', linewidth=1, label='投影方向ω')

        wu = w / np.linalg.norm(w)  # 转化为单位向量

        # 样本点投影（参考：https://blog.csdn.net/guyuealian/article/details/53954005）
        X0_project = np.dot(X0, np.dot(wu.T, wu))
        plt.scatter(X0_project[:, 0], X0_project[:, 1], c='k', s=15)
        for i in range(X0.shape[0]):
            plt.plot([X0[i, 0], X0_project[i, 0]], [X0[i, 1], X0_project[i, 1]], 'k--')

        X1_project = np.dot(X1, np.dot(wu.T, wu))
        plt.scatter(X1_project[:, 0], X1_project[:, 1], c='r', s=15)
        for i in range(X1.shape[0]):
            plt.plot([X1[i, 0], X1_project[i, 0]], [X1[i, 1], X1_project[i, 1]], 'r--')

        # 中心点投影
        u0_project = np.dot(u0, np.dot(wu.T, wu))
        plt.scatter(u0_project[:, 0], u0_project[:, 1], c='#696969', s=60)
        u1_project = np.dot(u1, np.dot(wu.T, wu))
        plt.scatter(u1_project[:, 0], u1_project[:, 1], c='#FF4500', s=60)

        ax.annotate(
            'u0投影点',
            xy=(u0_project[:, 0], u0_project[:, 1]),
            xytext=(u0_project[:, 0] - 0.25, u0_project[:, 1] + 0.16),
            size=13,
            va='center',
            ha='left',
            arrowprops=dict(
                arrowstyle='->',
                color='k'
            )
        )
        ax.annotate(
            'u1投影点',
            xy=(u1_project[:, 0], u1_project[:, 1]),
            xytext=(u1_project[:, 0] - 0.2, u1_project[:, 1] + 0.16),
            size=13,
            va='center',
            ha='left',
            arrowprops=dict(
                arrowstyle='->',
                color='r'
            )
        )

        plt.legend(loc='upper right')
        plt.axis("equal")  # 两坐标轴的单位刻度长度保存一致
        plt.savefig('3.5.png')
        plt.show()

        self.w = w
        self.u0 = u0
        self.u1 = u1

    def predict(self, X):
        project = np.dot(X, self.w.T)

        w_u0 = np.dot(self.w, self.u0.T)
        w_u1 = np.dot(self.w, self.u1.T)

        # 投影点到 w_u1的距离比到w_u0的距离近，即归为1的一类点
        return (np.abs(project - w_u1) < np.abs(project - w_u0)).astype(int)


data = pd.read_csv('../3.3/watermelon3_0_Ch.csv').values
X = data[:, 7:9].astype(float)
y = data[:, 9]
y[y == '是'] = 1
y[y == '否'] = 0
y = y.astype(int)
lda = LDA()
lda.fit(X, y)
y_predict = lda.predict(X).flatten()

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
y_clf_predict = clf.predict(X)
plt.figure(figsize=(15, 10))
plt.plot(np.arange(X.shape[0]), y, label='True')
plt.plot(np.arange(X.shape[0]), y_predict, label='Predict')
plt.plot(np.arange(X.shape[0]), y_clf_predict, label='Sklearn_Predict')
plt.legend()
plt.savefig('predict.png')
plt.show()
