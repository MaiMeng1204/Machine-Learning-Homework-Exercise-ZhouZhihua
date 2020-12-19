import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import PlotTree as pt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


def func(item):
    if item == '是':
        return 1
    else:
        return 0


class Node():
    def __init__(self):
        self.class_ = None
        self.attribute = None
        self.purity = None
        self.split_point = None
        self.child = {}
        self.is_leaf = True
        self.leaf_num = 1
        self.depth = 1


class DecisionTree():
    def __init__(self, criterion):
        self.criterion = criterion
        self.tree = None

    def info_entropy(self, y):
        '''
        计算信息熵
        输入：
        y：标签向量
        输出：
        entropy：信息熵
        '''
        p = y.value_counts() / len(y)  # 计算各类样本所占比例
        entropy = np.sum(-p * np.log2(p))
        return entropy

    def info_gain(self, x, y, EntD):
        '''
        求单个属性的信息增益
        输入：
        x：某个特征Series
        y：标签向量
        EntD：样本集信息熵
        输出：
        info_gain：list 信息增益,[划分点]
        '''
        attributes = x.unique()     # 所有的属性取值
        attri_sort = np.sort(attributes)    # 先排序
        Ta = ((attri_sort + np.roll(attri_sort, 1)) / 2)[1:]    # 候选划分点
        max_info_gain = -np.inf
        split_point = None
        for t in Ta:
            D1 = y[x <= t]
            D2 = y[x > t]
            attri_entropy = (len(D1) * self.info_entropy(D1) + len(D2) * self.info_entropy(D2)) / len(y)
            info_gain = EntD - attri_entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                split_point = t
        return [max_info_gain, split_point]

    def get_best_attribute_information_gain(self, X, y):
        '''
        通过信息增益获得最优划分属性
        输出：
        best_attribute 最优划分属性：str
        max_info_gain 信息增益,[划分点]：list
        '''
        attributes = X.columns
        best_attribute = None
        max_info_gain = [-np.inf]
        EntD = self.info_entropy(y)
        for attribute in attributes:
            info_gain = self.info_gain(X[attribute], y, EntD)
            if info_gain[0] > max_info_gain[0]:
                max_info_gain = info_gain
                best_attribute = attribute

        return best_attribute, max_info_gain

    def get_best_attribute(self, X, y):
        '''
        通过不同方式获取最优划分属性
        X：训练集
        y：标签向量
        '''
        if self.criterion == 'information gain':
            return self.get_best_attribute_information_gain(X, y)

    def generate_tree(self, X, y):
        node = Node()
        node.class_ = y.value_counts().index[0]

        if y.nunique() == 1 or X.empty:
            return node

        best_attribute, best_purity = self.get_best_attribute(X, y)
        node.attribute = best_attribute
        node.purity = best_purity[0]
        node.is_leaf = False
        node.leaf_num = 0

        node.split_point = best_purity[1]    # 划分点
        up_part = '> {:.3f}'.format(node.split_point)
        down_part = '<= {:.3f}'.format(node.split_point)

        X_up = X[X[best_attribute] > node.split_point]
        y_up = y[X[best_attribute] > node.split_point]
        X_down = X[X[best_attribute] <= node.split_point]
        y_down = y[X[best_attribute] <= node.split_point]
        node.child[up_part] = self.generate_tree(X_up, y_up)
        node.child[down_part] = self.generate_tree(X_down, y_down)
        node.leaf_num += (node.child[up_part].leaf_num + node.child[down_part].leaf_num)
        node.depth = max(node.child[up_part].depth, node.child[down_part].depth) + 1

        return node


data = pd.read_csv('E:\Python\机器学习\MachineLearning_Zhouzhihua_ProblemSets-master\data\watermelon3_0_Ch.csv', index_col=0)
X_discrete = data.iloc[:, :6].values
X_continues = data.iloc[:, 6:8].values
enc = OneHotEncoder(sparse=False)
X_onehot = enc.fit_transform(X_discrete)    # 将离散值转化为one-hot编码
X = np.append(X_onehot, X_continues, axis=1)
y = data.iloc[:, -1].map(func).to_numpy()
# 逻辑回归
lr = LogisticRegression()
lr.fit(X, y)
X = np.dot(X, lr.coef_.T) + lr.intercept_
X = pd.DataFrame(X, columns=['逻辑回归值'])
y = pd.Series(y)
y[y == 1] = '好瓜'
y[y == 0] = '坏瓜'
res = pd.concat([X, y], axis=1)
res.to_csv('逻辑回归结果.csv', encoding='gbk')
tree = DecisionTree(criterion='information gain')
tree.tree = tree.generate_tree(X, y)
pt.create_plot(tree.tree, '逻辑回归决策树')
