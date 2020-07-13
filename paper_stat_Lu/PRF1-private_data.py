import matplotlib.pyplot as plt
import numpy as np

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

pfli = []

app = ['Google', 'Twitter', 'Youtube', 'Outlook', 'Github', 'Facebook', 'Slack', 'Bing']

for i in range(0, 8):
    tpfn = sum(data[i])
    tp = data[i][i]
    tpfp = 0
    for j in range(0, 8):
        tpfp += data[j][i]
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = 2 * precision * recall / (precision + recall)
    pfli.append([precision, recall, f1])
# print("{0}\t{1}\t{2}".format(precision,recall,f1))


i = 0

# data to plot
n_groups = 8
precision = (pfli[i][0], pfli[i + 1][0], pfli[i + 2][0], pfli[i + 3][0], pfli[i + 4][0], pfli[i + 5][0], pfli[i + 6][0],
             pfli[i + 7][0])
recall = (pfli[i][1], pfli[i + 1][1], pfli[i + 2][1], pfli[i + 3][1], pfli[i + 4][1], pfli[i + 5][1], pfli[i + 6][1],
          pfli[i + 7][1])
f1 = (pfli[i][2], pfli[i + 1][2], pfli[i + 2][2], pfli[i + 3][2], pfli[i + 4][2], pfli[i + 5][2], pfli[i + 6][2],
      pfli[i + 7][2])
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.28
opacity = 0.7

rects1 = plt.bar(index, precision, bar_width, alpha=opacity, color='b', label='Frank')
rects2 = plt.bar(index + bar_width, recall, bar_width, alpha=opacity, color='g', label='Guido')
rects3 = plt.bar(index + 2 * bar_width, f1, bar_width, alpha=opacity, color='r', label='G')


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, '%d' % int(height * 100), fontsize=5, ha='center', va='bottom')
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, '%.2f' % (height * 1.0), fontsize=5, ha='center',
                va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
# plt.xlabel('Catagory')
plt.ylabel('Classification performance in Precision, Recall, and F1')
# plt.title('Scores by category')
plt.xticks(index + bar_width,
           (app[i], app[i + 1], app[i + 2], app[i + 3], app[i + 4], app[i + 5], app[i + 6], app[i + 7]))
plt.tight_layout()

plt.legend(['Precision', 'Recall', 'F1'], loc='lower right')
# plt.savefig("PRF1_"+str(i)+".jpg", dpi = 400)
plt.savefig('precision_recall_F1.pdf')
plt.show()
