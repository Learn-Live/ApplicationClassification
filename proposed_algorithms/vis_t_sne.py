# -*- coding: utf-8 -*-
"""
    visualize high-dimensions data by T-SNE

    refer to :
            1 http://alexanderfabisch.github.io/t-sne-in-scikit-learn.html
            2 http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
"""
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# from preprocess.data_preprocess import load_data, remove_special_labels
from numpy_load_and_arff import load_npy_data


def vis_high_dims_data_umap(X, y, show_label_flg=False):
    """

    :param X:  features
    :param y:  labels
    :param show_label_flg :
    :return:
    """
    # res_umap=umap.UMAP(n_neighbors=5,min_dist=0.3, metric='correlation').fit_transform(X,y)
    res_umap = umap.UMAP(n_neighbors=10, min_dist=0.3, spread=2, metric='correlation').fit_transform(X, y)

    if not show_label_flg:
        plt.figure(figsize=(10, 5))
        plt.scatter(res_umap[:, 0], res_umap[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10), alpha=0.7)
        plt.colorbar(ticks=range(10))
        plt.title('umap results')
        plt.show()
    else:
        plot_with_labels(X, y, res_umap, "UMAP", min_dist=20.0)



def vis_high_dims_data_t_sne(X, y, show_label_flg=False):
    """

    :param X:  features
    :param y:  labels
    :param show_label_flg :
    :return:
    """
    res_tsne = TSNE(n_components=2, verbose=2, learning_rate=1, n_iter=500, random_state=0).fit_transform(X, y)

    if not show_label_flg:
        plt.figure(figsize=(10, 5))
        plt.scatter(res_tsne[:, 0], res_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10), alpha=0.7)
        plt.colorbar(ticks=range(10))
        plt.title('tsne results')
        plt.show()
    else:
        plot_with_labels(X, y, res_tsne, "t-SNE", min_dist=20.0)


def vis_high_dims_data_pca(X, y, show_label_flg=False):
    """

    :param X:  features
    :param y:  labels
    :param show_label_flg :
    :return:
    """
    res_tsne = PCA(n_components=2, random_state=0).fit_transform(X, y)

    if not show_label_flg:
        plt.figure(figsize=(10, 5))
        plt.scatter(res_tsne[:, 0], res_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10), alpha=0.7)
        plt.colorbar(ticks=range(10))
        plt.title('pca results')
        plt.show()
    else:
        plot_with_labels(X, y, res_tsne, "pca", min_dist=20.0)


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


def plot_with_labels(X, y, X_embedded, name, min_dist=10.0):
    """
        plot with labels
    :param X:
    :param y:
    :param X_embedded: Fit X into an embedded space and return that transformed output.
    :param name: title
    :param min_dist: min distance
    :return:
    """
    import matplotlib
    from matplotlib.pyplot import figure, title, axes, setp, subplots_adjust, scatter, cm
    import numpy as np
    # Plotting function
    matplotlib.rc('font', **{'family': 'sans-serif',
                             'weight': 'bold',
                             'size': 18})
    matplotlib.rc('text', **{'usetex': True})

    fig = figure(figsize=(10, 10))
    ax = axes(frameon=False)
    title("\\textbf{MNIST dataset} -- Two-dimensional "
          "embedding of 70,000 handwritten digits with %s" % name)
    setp(ax, xticks=(), yticks=())
    subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    scatter(X_embedded[:, 0], X_embedded[:, 1],
            c=y, marker="x")

    if min_dist is not None:
        from matplotlib import offsetbox
        shown_images = np.array([[15., 15.]])
        indices = np.arange(X_embedded.shape[0])
        np.random.shuffle(indices)
        for i in indices[:5000]:
            dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            shown_images = np.r_[shown_images, [X_embedded[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i].reshape(28, 28),
                                      cmap=cm.gray_r), X_embedded[i])
            ax.add_artist(imagebox)
    plt.show()


def change_labels(y):
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = 'a'
        elif y[i] == 1:
            y[i] = 'b'
        elif y[i] == 2:
            y[i] = 'c'
        elif y[i] == 3:
            y[i] = 'd'
        else:
            pass


if __name__ == '__main__':
    demo_flg = False
    if demo_flg:
        demo_t_sne()
    else:
        input_file = '../input_data/trdata-8000B_payload.npy'
        input_file = '../input_data/trdata-8000B_header_payload_20190326.npy'
        session_size=8000
        X, y = load_npy_data(input_file,session_size)
        # X = list(map(lambda t: t[:session_size], X))
        y = list(map(lambda t: int(float(t)), y))
        # change_labels(y)
        cntr = Counter(y)
        print('X: ', len(X), ' y:', sorted(cntr.items()))
        # vis_high_dims_data_pca(X, y, show_label_flg=False)
        # vis_high_dims_data_t_sne(X, y, show_label_flg=False)
        vis_high_dims_data_umap(X, y, show_label_flg=False)
