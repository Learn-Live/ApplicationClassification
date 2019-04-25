import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# plt.gcf().subplots_adjust(left=0.5, bottom=0.25, wspace=0.3, hspace=0.3)

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

data = [[66, 0, 0, 0, 1, 1, 0, 2],
        [0, 65, 0, 1, 3, 0, 2, 0],
        [0, 0, 52, 0, 0, 0, 0, 7],
        [3, 0, 0, 64, 0, 0, 0, 0],
        [0, 0, 0, 0, 66, 0, 6, 1],
        [0, 0, 0, 0, 2, 74, 4, 0],
        [0, 0, 0, 0, 7, 1, 60, 0],
        [0, 0, 3, 1, 0, 1, 1, 66]]


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(7, 5)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                # annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                annot[i, j] = '%d' % (c)
            # elif c == 0:
            #    annot[i, j] = ''
            else:
                # annot[i, j] = '%.1f%%\n%d' % (p, c)
                annot[i, j] = '%d' % (c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual labels'
    cm.columns.name = 'Predicted labels'
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots()
    # sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues")
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    # plt.savefig(filename, dpi = 600)
    # plt.tight_layout(.5)
    plt.savefig(filename)  # should use before plt.show()
    plt.show()


app = ['Google', 'Twitter', 'Youtube', 'Outlook', 'Github', 'Gacebook', 'Slack', 'Bing']
ypred = []
ytrue = []

for i in range(0, 8):
    for j in range(0, 8):
        num = data[i][j]

        for t in range(0, num):
            ypred.append(app[j])
            ytrue.append(app[i])
cm_analysis(ytrue, ypred, "confmatrix.pdf", app)
