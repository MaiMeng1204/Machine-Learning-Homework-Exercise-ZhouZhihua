import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import PlotTree as pt


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

    def generate_tree(self, X, y):
        node = Node()
        if y.nunique() == 1:
            node.class_ = y.iloc[0]
            return node

        if X.empty:
            node.class_ = y.value_counts().index[0]
            return node

        node.is_leaf = False
        node.leaf_num = 0
        node.attribute, node.purity = self.get_best_split_attribute(X, y)

        attribute_values = X[node.attribute].unique()
        X_child = X.drop(node.attribute, axis=1)
        max_depth = 0
        for attribute_value in attribute_values:
            mask = X[node.attribute] == attribute_value
            X_attribute_value = X_child[mask]
            y_attribute_value = y[mask]
            node.child[attribute_value] = self.generate_tree(X_attribute_value, y_attribute_value)
            if node.child[attribute_value].depth > max_depth:
                max_depth = node.child[attribute_value].depth
            node.leaf_num += node.child[attribute_value].leaf_num
        node.depth += max_depth

        return node


data = pd.read_csv('E:\Python\机器学习\MachineLearning_Zhouzhihua_ProblemSets-master\data\watermelon2_0_Ch.txt', index_col=0)
train = data.iloc[[1, 2, 3, 6, 7, 10, 14, 15, 16], :]
test = data.iloc[[4, 5, 8, 9, 11, 12, 13], :]
X_train = train.iloc[:, :6]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :6]
y_test = test.iloc[:, -1]

tree = DecisionTree(criterion='gini index', pruning=None)
tree.tree = tree.generate_tree(X_train, y_train)
pt.create_plot(tree.tree, '4.4')
