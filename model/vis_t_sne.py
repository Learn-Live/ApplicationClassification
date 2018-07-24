# -*- coding: utf-8 -*-
"""
    visualize high-dimensions data by T-SNE

"""
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from preprocess.data_preprocess import load_data


def vis_high_dims_data_t_sne(X, y):
    """

    :param X:  features
    :param y:  labels
    :return:
    """
    res_tsne = TSNE(n_components=3, verbose=2, learning_rate=100.0, n_iter=1000).fit_transform(X, y)

    plt.figure(figsize=(10, 5))
    plt.scatter(res_tsne[:, 0], res_tsne[:, 1], c=y)
    plt.title('results')
    plt.show()


def demo_t_sne():
    """
        display iris_data by TSNE
    :return:
    """

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    iris = load_iris()
    X_tsne = TSNE(learning_rate=100, n_components=3, perplexity=40, verbose=2).fit_transform(iris.data)
    X_pca = PCA().fit_transform(iris.data)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
    plt.title('TSNE')
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
    plt.title('PCA')
    plt.show()


if __name__ == '__main__':
    demo_flg = False
    if demo_flg:
        demo_t_sne()
    else:
        input_file = '../results/skype_784Bytes_train.csv'
        X, y = load_data(input_file)
        y = list(map(lambda t: int(float(t)), y))
        cntr = Counter(y)
        print('X: ', len(X), ' y:', sorted(cntr.items()))
        vis_high_dims_data_t_sne(X, y)
