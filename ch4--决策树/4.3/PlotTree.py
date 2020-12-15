import matplotlib.pyplot as plt


decision_node = dict(boxstyle='round,pad=0.3', fc='#FAEBD7')
leaf_node = dict(boxstyle='round,pad=0.3', fc='#F4A460')


def plot_node(node_text, center_xy, parent_xy, node_type, ax):
    ax.annotate(
        node_text,
        xy=[parent_xy[0], parent_xy[1] - 0.02],
        xycoords='axes fraction',
        xytext=center_xy,
        textcoords='axes fraction',
        va='center',
        ha='center',
        size=15,
        bbox=node_type,
        arrowprops=dict(arrowstyle='<-')
    )


def plot_mid_text(text, center_xy, parent_xy, ax):
    '''
    绘制节点间连接文本
    '''
    x_mid = (parent_xy[0] + center_xy[0]) / 2
    y_mid = (parent_xy[1] + center_xy[1]) / 2
    ax.text(x_mid, y_mid, text, fontdict=dict(size=12))


def plot_tree(tree, parent_xy, node_text, ax):
    '''
    绘制决策树

    Parameters：
    tree：树节点
    parent_xy：父节点坐标
    node_text：节点文本（划分属性）
    ax：画图对象

    Return: None
    '''
    global x_off
    global y_off
    global total_leaf_num
    global total_depth

    leaf_num = tree.leaf_num
    center_xy = (x_off + ((1 + leaf_num) / 2) / total_leaf_num, y_off)    # 第一个坐标是使x_off落在中点位置，即 (x/2) / total

    plot_mid_text(node_text, center_xy, parent_xy, ax)

    # 深度为1，表示只有一个节点
    if tree.depth == 1:
        plot_node(tree.class_, center_xy, parent_xy, leaf_node, ax)
        return

    plot_node(tree.attribute, center_xy, parent_xy, decision_node, ax)

    y_off -= 1 / total_depth    # 画下一层节点
    for key, value in tree.child.items():
        if value.is_leaf:
            x_off += 1 / total_leaf_num
            plot_node(value.class_, (x_off, y_off), center_xy, leaf_node, ax)
            plot_mid_text(key, (x_off, y_off), center_xy, ax)
        else:
            plot_tree(value, center_xy, key, ax)    # center_xy作为下一个节点的parent_xy
    y_off += 1 / total_depth    # 画完当前节点的子节点后，回到父节点那一层


def create_plot(tree, filename):
    '''
    创建画布
    '''
    global x_off
    global y_off
    global total_leaf_num
    global total_depth

    total_leaf_num = tree.leaf_num
    total_depth = tree.depth
    x_off = -0.5 / total_leaf_num   # x_off, y_off都是按占比定义的，可以在plot_node的xycoords='axes fraction'看出
    y_off = 1

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xticks([])  # 隐藏坐标轴刻度
    ax.set_yticks([])
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    plot_tree(tree, (0.5, 1), '', ax)
    fig.savefig(filename + '.jpg')
    plt.show()
