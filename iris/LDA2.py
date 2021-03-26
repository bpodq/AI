import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA_dimensionality(X, y, k):
    '''
    X为数据集，y为label，k为目标维数
    '''
    label_ = list(set(y))

    X_classify = {}

    for label in label_:
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == label])
        X_classify[label] = X1

    mu = np.mean(X, axis=0)
    mu_classify = {}

    for label in label_:
        mu1 = np.mean(X_classify[label], axis=0)
        mu_classify[label] = mu1

    #St = np.dot((X - mu).T, X - mu)

    Sw = np.zeros((len(mu), len(mu)))  # 计算类内散度矩阵
    for i in label_:
        Sw += np.dot((X_classify[i] - mu_classify[i]).T,
                     X_classify[i] - mu_classify[i])

    # Sb=St-Sw

    Sb = np.zeros((len(mu), len(mu)))  # 计算类内散度矩阵
    for i in label_:
        Sb += len(X_classify[i]) * np.dot((mu_classify[i] - mu).reshape(
            (len(mu), 1)), (mu_classify[i] - mu).reshape((1, len(mu))))

    eig_vals, eig_vecs = np.linalg.eig(
        np.linalg.inv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵

    sorted_indices = np.argsort(eig_vals)
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]  # 提取前k个特征向量

    return topk_eig_vecs


if '__main__' == __name__:

    iris = load_iris()
    X = iris.data
    y = iris.target

    W = LDA_dimensionality(X, y, 2)
    X_new = np.dot(X, W)
    plt.figure(1)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)

    # 与sklearn中的LDA函数对比
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_new = lda.transform(X)
    print(X_new)
    plt.figure(2)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)

    plt.show()

