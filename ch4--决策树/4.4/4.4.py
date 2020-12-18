import numpy as np
import pandas as pd
import sys
sys.path.append('..')
# import PlotTree as pt


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
    def __init__(self, criterion, pruning):
        self.criterion = criterion
        self.pruning = pruning
        self.tree = None

    def gini(self, y):
        p = y.value_counts() / len(y)
        gini = 1 - np.sum(np.power(p, 2))
        return gini

    def gini_index(self, x, y):
        attribute_values = x.unique()
        gini_index = 0
        for attribute_value in attribute_values:
            Dv = y[x == attribute_value]
            gini_index += len(Dv) / len(y) * self.gini(Dv)
        return gini_index

    def get_best_split_attribute_gini_index(self, X, y):
        attributes = X.columns
        min_gini_index = np.inf
        best_split_attribute = None
        for attribute in attributes:
            gini_index = self.gini_index(X[attribute], y)
            if gini_index < min_gini_index:
                min_gini_index = gini_index
                best_split_attribute = attribute
        return best_split_attribute, min_gini_index

    def get_best_split_attribute(self, X, y):
        if self.criterion == 'gini index':
            return self.get_best_split_attribute_gini_index(X, y)

    def prepruning(self, node, X_train, y_train, X_test, y_test):
        '''
        计算剪枝后的准确率
        '''
        attribute = node.attribute
        y_train_split = y_train.groupby(X_train[attribute]).apply(lambda x: x.value_counts().index[0])  # 划分后每个节点的类别取值
        correct_y = y_test.groupby(X_test[attribute]).apply(lambda x: np.sum(x == y_train_split[x.name]))   # 分类正确的个数
        accuracy = np.sum(correct_y) / len(y_test)
        return accuracy

    def generate_tree(self, X_train, y_train, X_test, y_test):
        node = Node()
        node.class_ = y_train.value_counts().index[0]

        if y_train.nunique() == 1 or X_train.empty:
            return node

        node.attribute, node.purity = self.get_best_split_attribute(X_train, y_train)

        # 不剪枝的准确率
        current_accuracy = np.mean(y_test == node.class_)
        # 预剪枝
        if self.pruning == 'prepruning':
            pruning_accuracy = self.prepruning(node, X_train, y_train, X_test, y_test)
        if current_accuracy >= pruning_accuracy:
            return node

        attribute_values = X_train[node.attribute].unique()
        X_child = X_train.drop(node.attribute, axis=1)
        max_depth = 0
        for attribute_value in attribute_values:
            mask = X_train[node.attribute] == attribute_value
            X_attribute_value = X_child[mask]
            y_attribute_value = y_train[mask]
            node.child[attribute_value] = self.generate_tree(X_attribute_value, y_attribute_value)
            if node.child[attribute_value].depth > max_depth:
                max_depth = node.child[attribute_value].depth
            node.leaf_num += node.child[attribute_value].leaf_num
        node.depth += max_depth

        return node


data = pd.read_csv('E:\Python\机器学习\MachineLearning_Zhouzhihua_ProblemSets-master\data\watermelon2_0_Ch.txt', index_col=0)
train = data.iloc[np.array([1, 2, 3, 6, 7, 10, 14, 15, 16, 17]) - 1, :]
test = data.iloc[np.array([4, 5, 8, 9, 11, 12, 13]) - 1, :]
X_train = train.iloc[:, :6]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :6]
y_test = test.iloc[:, -1]

tree = DecisionTree(criterion='gini index', pruning='prepruning')
tree.tree = tree.generate_tree(X_train, y_train, X_test, y_test)
pt.create_plot(tree.tree, '预剪枝决策树')
