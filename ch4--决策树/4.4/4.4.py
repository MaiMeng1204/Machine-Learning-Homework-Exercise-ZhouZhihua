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

    def pruning_accuracy(self, node, X_train, y_train, X_test, y_test):
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
            pruning_accuracy = self.pruning_accuracy(node, X_train, y_train, X_test, y_test)
            if current_accuracy >= pruning_accuracy:
                return node

        attribute_values = X_train[node.attribute].unique()
        X_train_child = X_train.drop(node.attribute, axis=1)
        max_depth = 0
        node.leaf_num = 0
        for attribute_value in attribute_values:
            train_mask = X_train[node.attribute] == attribute_value
            test_mask = X_test[node.attribute] == attribute_value
            node.child[attribute_value] = self.generate_tree(X_train_child[train_mask], y_train[train_mask], X_test.loc[test_mask], y_test.loc[test_mask])
            if node.child[attribute_value].depth > max_depth:
                max_depth = node.child[attribute_value].depth
            node.leaf_num += node.child[attribute_value].leaf_num
        node.depth += max_depth
        node.is_leaf = False

        return node

    def post_pruning(self, node, X_train, y_train, X_test, y_test):
        '''
        后剪枝决策树
        '''
        # 若为叶节点，直接返回
        if node.is_leaf:
            return

        # 重设叶节点个数与深度
        node.leaf_num = 0
        node.depth = 1
        max_depth = 0

        # 递归
        for attribute_value, child in node.child.items():
            child_X_train = X_train[X_train[node.attribute] == attribute_value]
            child_y_train = y_train[X_train[node.attribute] == attribute_value]
            child_X_test = X_test[X_test[node.attribute] == attribute_value]
            child_y_test = y_test[X_test[node.attribute] == attribute_value]
            self.post_pruning(child, child_X_train, child_y_train, child_X_test, child_y_test)

            node.leaf_num += child.leaf_num
            if child.depth > max_depth:
                max_depth = child.depth
        node.depth += max_depth

        # 不是叶节点，开始后剪枝
        if not node.is_leaf:
            # 不剪枝的准确率
            current_accuracy = np.mean(y_test == node.class_)
            pruning_accuracy = self.pruning_accuracy(node, X_train, y_train, X_test, y_test)
            # 剪枝
            if pruning_accuracy <= current_accuracy:
                node.class_ = y_train.value_counts().index[0]
                node.attribute = None
                node.purity = None
                node.child = {}
                node.is_leaf = True
                node.leaf_num = 1
                node.depth = 1


data = pd.read_csv('E:\Python\机器学习\MachineLearning_Zhouzhihua_ProblemSets-master\data\watermelon2_0_Ch.txt', index_col=0)
train = data.iloc[np.array([1, 2, 3, 6, 7, 10, 14, 15, 16, 17]) - 1, :]
test = data.iloc[np.array([4, 5, 8, 9, 11, 12, 13]) - 1, :]
X_train = train.iloc[:, :6]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :6]
y_test = test.iloc[:, -1]

tree = DecisionTree(criterion='gini index', pruning='postpruning')
tree.tree = tree.generate_tree(X_train, y_train, X_test, y_test)
if tree.pruning == 'postpruning':
    tree.post_pruning(tree.tree, X_train, y_train, X_test, y_test)
pt.create_plot(tree.tree, '后剪枝决策树')
