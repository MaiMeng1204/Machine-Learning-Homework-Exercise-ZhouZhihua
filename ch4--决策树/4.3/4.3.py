import numpy as np
import pandas as pd


class Node():
    def __init__(self):
        self.class_ = None  # 节点所属类别
        self.attribute = None   # 节点划分属性
        self.purity = None  # 节点纯度（信息增益、增益率等）
        self.split_point = None     # 划分点
        self.attribute_index = None     # 划分属性索引
        self.child = {}     # 子节点字典，key为属性取值，value为node
        self.is_leaf = False    # 是否为叶节点
        self.leaf_num = None    # 叶子节点数量
        self.depth = -1  # 节点深度


class DecisionTree():
    def __init__(self, criterion='information gain', pruning=None):
        self.criterion = criterion
        self.pruning = pruning
        self.tree = None    # 树

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
        # 离散变量
        if x.dtypes == 'O':
            attri_entropy = 0
            for attri in attributes:
                Dv = y[x == attri]  # 提取样本子集
                attri_entropy += (len(Dv) / len(y) * self.info_entropy(Dv))     # 信息增益
            info_gain = EntD - attri_entropy
            return [info_gain]

        # 连续变量
        else:
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
        '''
        生成决策树
        输入：
        X：训练集
        y：标签向量
        输出：
        node：树节点
        '''
        node = Node()   # 建立节点
        node.leaf_num = 0
        # 属于同一类别
        if y.nunique() == 1:
            node.is_leaf = True     # 叶节点
            node.class_ = y.values[0]
            node.depth = 1  # 深度为1
            node.leaf_num += 1
            return node

        # 样本集为空
        if X.empty:
            node.is_leaf = True
            node.class_ = y.value_counts().index[0]  # 类别标记为D中样本最多的类
            node.depth = 1
            node.leaf_num += 1
            return node

        best_attribute, best_purity = self.get_best_attribute(X, y)
        print(best_attribute)
        node.attribute = best_attribute
        node.purity = best_purity[0]
        node.attribute_index = X.columns.get_loc(node.attribute)

        # 离散值
        if len(best_purity) == 1:
            attribute_values = X[best_attribute].unique()
            X_child = X.drop(best_attribute, axis=1)    # 删掉划分列

            max_depth = 0
            for attribute_value in attribute_values:
                node.child[attribute_value] = self.generate_tree(
                    X_child[X[best_attribute] == attribute_value],
                    y[X[best_attribute] == attribute_value]
                )
                # 记录子树最大高度
                if node.child[attribute_value].depth > max_depth:
                    max_depth = node.child[attribute_value].depth
                node.leaf_num += node[attribute_value].leaf_num
            node.depth = max_depth + 1

        # 连续值
        else:
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
X = data.iloc[:, :8]
y = data.iloc[:, -1]
tree = DecisionTree(criterion='information gain')
tree.tree = tree.generate_tree(X, y)
