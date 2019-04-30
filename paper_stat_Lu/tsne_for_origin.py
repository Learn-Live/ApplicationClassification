from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random, time
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})


def vis_high_dims_data_t_sne(X, y, show_label_flg=False):
    res_tsne = TSNE(n_components=2, verbose=2, learning_rate=1, n_iter=500, random_state=0).fit_transform(X, y)
    plt.figure(figsize=(10, 5))
    plt.scatter(res_tsne[:, 0], res_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10), alpha=0.7)
    plt.colorbar(ticks=range(10))
    plt.title('tsne results')
    plt.savefig("t_SNE_origin.jpg", dpi=400)
    plt.savefig('original_space_data.pdf')
    plt.show()


def get_data_from_file(input_file):
    SAMPLE = 700
    marco = 3000
    resample = True
    removelist = [3, 5]  # remove y label
    catenumber = 8
    a = np.load(input_file)[1:]
    trainY = (a[:, 0]).reshape(-1, 1)
    trainX = (a[:, 1])
    truetrainX = []
    truetrainY = []
    for i in range(0, len(trainX)):
        Xli = trainX[i][0]
        Yli = trainY[i][0]
        containinfo = False
        # remove certain data
        if Yli in removelist:
            continue
        elif (Yli == 4):
            Yli = 3
        elif (Yli > 5):
            Yli -= 2
        for i in Xli:
            if i != 0:
                containinfo = True
                break
        if (containinfo == True):
            truetrainX.append(Xli)
            truetrainY.append(Yli)
    truetrainX = np.asarray(truetrainX)
    truetrainY = np.asarray(truetrainY)
    ss = StandardScaler()
    truetrainX = ss.fit_transform(truetrainX[:, :marco])
    # resample
    listdir = {}
    for i in range(0, len(truetrainY)):
        if truetrainY[i] not in listdir:
            listdir.update({truetrainY[i]: [i]})
        else:
            listdir[truetrainY[i]].append(i)
    actualdir = {}
    for i in range(0, catenumber):
        if i in listdir:
            thelist = listdir[i]
        else:
            thelist = []
        if (len(thelist) > SAMPLE):
            actualdir.update({i: random.sample(thelist, SAMPLE)})  # sample 500
        else:
            actualdir.update({i: thelist})
    listdir = {}
    dic = {}
    truetruetrainX = []
    truetruetrainY = []
    for i in range(0, len(truetrainY)):
        if i not in actualdir[truetrainY[i]]:
            continue
        truetruetrainX.append(truetrainX[i])
        truetruetrainY.append(truetrainY[i])
    X = np.asarray(truetruetrainX)
    Y = np.asarray(truetruetrainY)

    return X, Y


import umap
import numpy as np


def vis_high_dims_data_umap(X, y, show_label_flg=False):
    """

    :param X:  features
    :param y:  labels
    :param show_label_flg :
    :return:
    """
    # res_umap=umap.UMAP(n_neighbors=5,min_dist=0.3, metric='correlation').fit_transform(X,y)
    res_umap = umap.UMAP(n_neighbors=40, min_dist=0.9, metric='correlation', random_state=42).fit_transform(X, y)

    if not show_label_flg:
        # plt.figure(figsize=(10, 5))
        fig, ax = plt.subplots(figsize=(12, 7))
        # plt.setp(ax, xticks=[], yticks=[])

        cax = plt.scatter(res_umap[:, 0], res_umap[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 8), alpha=0.8)
        # cbar = fig.colorbar(mappable=cax, ax=ax, orientation='horizontal')
        cbar = fig.colorbar(mappable=cax, ax=ax)
        cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7])
        cbar.set_ticklabels(
            ['Google', 'Twitter', 'Outlook', 'Youtube', 'Github', 'Facebook', 'Slack', 'Bing'])  # horizontal colorbar
        # new8 = {0:'google',1:'twitter',2:'outlook',3:'youtube',4:'github',5:'facebook',6:'slack',7:'bing'}
        plt.setp(ax, xticks=[], yticks=[])
        # plt.title('umap results')
        plt.savefig('original_space_data.pdf')
        plt.show()
    else:
        pass
        # plot_with_labels(X, y, res_umap, "UMAP", min_dist=2.0)


np.random.seed(42)
from sklearn.preprocessing import StandardScaler

input_file = '../input_data/newapp_10220_pt.npy'
X, Y = get_data_from_file(input_file)
ss = StandardScaler()
X = ss.fit_transform(X)
vis_high_dims_data_umap(X, Y)
