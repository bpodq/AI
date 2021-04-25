# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, discriminant_analysis


def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train, y_train,
                            test_size=0.25, random_state=0, stratify=y_train)  # stratify分层


def plot_LDA(converted_X, y):
    '''
    绘制经过 LDA 转换后的数据
    :param converted_X: 经过 LDA转换后的样本集(150,3)降维后的表示方法
    :param y: 样本集的标记(150,1)
    :return:  None
    '''

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = 'rgb'
    markers = 'o*s'

    for target, color, marker in zip([0, 1, 2], colors, markers):  # zip()方法用在for循环中，支持并行迭代
        pos = (y == target).ravel()  # 由标签转换为bool矩阵，分成不同标签类型的矩阵(150,) ravel平铺用于拆分数据
        X = converted_X[pos, :]  # 取出对应pos中True的元素，目的是将降维后的数据拆分为三类X(50,3)---》50行数据正好下面可以取每一列共50

        # 三维数据
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, marker=marker,  # X[:,0]取出第一列的数据
                   label="Label %d" % target)

    ax.legend(loc="best")
    fig.suptitle("Iris After LDA")
    plt.show()


def run_plot_LDA():
    '''
    执行 plot_LDA 。其中数据集来自于 load_data() 函数
    :return: None
    '''
    # X_train(112,4)  X_test(38,4) y_train(112,) y_test(38,)
    X_train, X_test, y_train, y_test = load_data()

    # X(150,4)
    X = np.vstack((X_train, X_test))  # 考虑到lda的fit接受的参数问题,这里需要组合X_Train和X_test

    # Y(150,1)  原本维度为1,所以要先reshape
    Y = np.vstack((y_train.reshape(y_train.size, 1), y_test.reshape(y_test.size, 1)))  # 组合y_train和y_test#size行1列
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X, Y.ravel())  # lda.fit(X,Y)就会出警告

    # 权值lda.coef_(3, 4)  lda.intercept_(3,)
    # converted_X(150,3) 对数据降维了
    converted_X = np.dot(X, np.transpose(lda.coef_)) + lda.intercept_  # X*权值+偏置b  就是输出值

    # print(converted_X)
    plot_LDA(converted_X, Y)


if __name__ == '__main__':
    run_plot_LDA()

# 为什么最终结果是在3维空间一条线上？

