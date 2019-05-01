import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

# cnn_data = [[66, 0, 0, 0, 1, 1, 0, 2],  # data for cnn
#             [0, 65, 0, 1, 3, 0, 2, 0],
#             [0, 0, 52, 0, 0, 0, 0, 7],
#             [3, 0, 0, 64, 0, 0, 0, 0],
#             [0, 0, 0, 0, 66, 0, 6, 1],
#             [0, 0, 0, 0, 2, 74, 4, 0],
#             [0, 0, 0, 0, 7, 1, 60, 0],
#             [0, 0, 3, 1, 0, 1, 1, 66]]
#
# dt_data = [[66, 0, 0, 0, 2, 1, 0, 1],
#            [1, 62, 0, 0, 4, 0, 4, 0],
#            [0, 0, 72, 0, 0, 1, 0, 4],
#            [1, 1, 0, 61, 0, 3, 0, 1],
#            [2, 8, 0, 2, 50, 2, 7, 2],
#            [0, 2, 0, 0, 1, 77, 0, 0],
#            [0, 0, 0, 1, 3, 2, 61, 1],
#            [0, 0, 3, 2, 1, 0, 0, 48]]

cnndata = [[66, 0, 0, 0, 1, 1, 0, 2],  # data for cnn
           [0, 65, 0, 1, 3, 0, 2, 0],
           [0, 0, 52, 0, 0, 0, 0, 7],
           [3, 0, 0, 64, 0, 0, 0, 0],
           [0, 0, 0, 0, 66, 0, 6, 1],
           [0, 0, 0, 0, 2, 74, 4, 0],
           [0, 0, 0, 0, 7, 1, 60, 0],
           [0, 0, 3, 1, 0, 1, 1, 66]]

dtdata = [[66, 0, 0, 0, 2, 1, 0, 1],
          [1, 62, 0, 0, 4, 0, 4, 0],
          [0, 0, 72, 0, 0, 1, 0, 4],
          [1, 1, 0, 61, 0, 3, 0, 1],
          [2, 8, 0, 2, 50, 2, 7, 2],
          [0, 2, 0, 0, 1, 77, 0, 0],
          [0, 0, 0, 1, 3, 2, 61, 1],
          [0, 0, 3, 2, 1, 0, 0, 48]]

lrdata = [[66, 0, 0, 0, 2, 1, 1, 0],
          [0, 60, 2, 1, 3, 1, 4, 0],
          [0, 0, 41, 1, 4, 0, 6, 9],
          [1, 1, 1, 58, 0, 0, 3, 3],
          [0, 2, 5, 2, 38, 0, 17, 9],
          [0, 2, 2, 1, 6, 63, 4, 2],
          [0, 6, 0, 3, 15, 2, 38, 4],
          [0, 1, 19, 0, 5, 1, 6, 38]]

GaussianNBdata = [[52, 0, 12, 0, 1, 3, 0, 2],
                  [1, 32, 2, 1, 0, 4, 1, 30],
                  [0, 6, 11, 2, 0, 0, 1, 41],
                  [0, 1, 1, 41, 0, 17, 0, 7],
                  [2, 7, 0, 1, 0, 4, 3, 56],
                  [2, 5, 0, 0, 0, 40, 1, 32],
                  [7, 5, 2, 1, 2, 5, 2, 44],
                  [6, 5, 0, 0, 0, 1, 0, 58]]

KNNdata = [[65, 0, 1, 1, 0, 1, 1, 1],
           [0, 60, 3, 1, 0, 1, 2, 4],
           [0, 0, 40, 2, 7, 0, 2, 10],
           [1, 1, 4, 57, 1, 1, 1, 1],
           [0, 2, 13, 0, 42, 0, 8, 8],
           [1, 5, 9, 0, 5, 52, 5, 3],
           [1, 3, 22, 4, 8, 0, 27, 3],
           [5, 1, 32, 0, 10, 1, 3, 18]]

svmdata = [[65, 0, 0, 1, 2, 1, 0, 1],
           [0, 60, 3, 1, 2, 0, 5, 0],
           [0, 1, 43, 1, 3, 0, 2, 11],
           [2, 1, 6, 55, 0, 0, 1, 2],
           [0, 4, 4, 1, 45, 0, 12, 7],
           [0, 2, 1, 3, 7, 59, 6, 2],
           [1, 6, 4, 2, 17, 1, 35, 2],
           [2, 1, 18, 0, 6, 0, 7, 36]]

test_results_dict = {'NB': GaussianNBdata, "SVM": svmdata, 'KNN': KNNdata, 'LR': lrdata, 'DT': dtdata, 'CNN': cnndata}
app = ['Google', 'Twitter', 'Youtube', 'Outlook', 'Github', 'Facebook', 'Slack', 'Bing']

# pfli = []
# for i in range(0, 8):
#     tpfn = sum(dt_data[i])
#     tp = dt_data[i][i]
#     tpfp = 0
#     for j in range(0, 8):
#         tpfp += dt_data[j][i]
#     precision = tp / tpfp
#     recall = tp / tpfn
#     f1 = 2 * precision * recall / (precision + recall)
#     pfli.append([precision, recall, f1])
# # print("{0}\t{1}\t{2}".format(precision,recall,f1))
#
#
# dtpfli = []
# for i in range(0, 8):
#     tpfn = sum(cnn_data[i])
#     tp = cnn_data[i][i]
#     tpfp = 0
#     for j in range(0, 8):
#         tpfp += cnn_data[j][i]
#     precision = tp / tpfp
#     recall = tp / tpfn
#     f1 = 2 * precision * recall / (precision + recall)
#     dtpfli.append([precision, recall, f1])
# # print("{0}\t{1}\t{2}".format(precision,recall,f1))

f1_dict = {}
for idx, (key, data) in enumerate(test_results_dict.items()):
    f1_lst = []
    for i in range(0, 8):
        tpfn = sum(data[i])
        tp = data[i][i]
        tpfp = 0
        for j in range(0, 8):
            tpfp += data[j][i]
        precision = tp / tpfp
        recall = tp / tpfn
        if precision + recall == 0:
            f1 = 0.0
            print(f'{key,i}')
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_lst.append([precision, recall, f1])
    f1_dict[key] = f1_lst
# print("{0}\t{1}\t{2}".format(precision,recall,f1))


i = 0

# data to plot
n_groups = 8

# f1 = (pfli[i][2], pfli[i + 1][2], pfli[i + 2][2], pfli[i + 3][2], pfli[i + 4][2], pfli[i + 5][2], pfli[i + 6][2],
#       pfli[i + 7][2])
# dtf1 = (dtpfli[i][2], dtpfli[i + 1][2], dtpfli[i + 2][2], dtpfli[i + 3][2], dtpfli[i + 4][2], dtpfli[i + 5][2],
#         dtpfli[i + 6][2], dtpfli[i + 7][2])
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
print(index)
bar_width = 0.15
opacity = 1

# # test_results_dict={'NB':GaussianNBdata,"SVM":svmdata,'KNN':KNNdata, 'LR':lrdata,'DT':dtdata,'CNN':cnndata}
key = 'NB'
f1 = list(map(lambda x: x[-1], f1_dict[key]))
rects1 = plt.bar(index, f1, bar_width, alpha=opacity, color='tab:brown', label='Frank')
key = 'KNN'
f1 = list(map(lambda x: x[-1], f1_dict[key]))
rects2 = plt.bar(index + (2 - 1) * bar_width, f1, bar_width, alpha=opacity, color='tab:green', label='Frank2')
key = 'SVM'
f1 = list(map(lambda x: x[-1], f1_dict[key]))
rects3 = plt.bar(index + (3 - 1) * bar_width, f1, bar_width, alpha=opacity, color='m', label='Frank3')
key = 'LR'
f1 = list(map(lambda x: x[-1], f1_dict[key]))
rects4 = plt.bar(index + (4 - 1) * bar_width, f1, bar_width, alpha=opacity, color='c', label='Frank4')
key = 'DT'
f1 = list(map(lambda x: x[-1], f1_dict[key]))
rects5 = plt.bar(index + (5 - 1) * bar_width, f1, bar_width, alpha=opacity, color='b', label='Frank5')
key = 'CNN'
f1 = list(map(lambda x: x[-1], f1_dict[key]))
rects6 = plt.bar(index + (6 - 1) * bar_width, f1, bar_width, alpha=opacity, color='r', label='Frank6')

# rects1 = plt.bar(index , f1, bar_width,alpha=opacity,color='b')
# rects2 = plt.bar(index + bar_width, dtf1, bar_width,alpha=opacity,color='r')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, '%d' % int(height * 100), fontsize=5, ha='center', va='bottom')
        ax.text(rect.get_x() + rect.get_width() / 2.3, 1.022 * height, '%.2f' % (height * 1.0), fontsize=5, ha='center',
                va='bottom')


# for i in range(6):
# autolabel(rects+i)
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
# plt.xlabel('Catagory')
plt.ylabel('F1')
# plt.title('F1 Scores by category')
plt.xticks(index + 2.5 * bar_width,
           (app[i], app[i + 1], app[i + 2], app[i + 3], app[i + 4], app[i + 5], app[i + 6], app[i + 7]))
plt.tight_layout()

# plt.legend([key for key in test_results_dict.keys() if key !='NB'], loc='lower right')
plt.legend(['NB', 'KNN', 'SVM', 'LR', 'DT', '1D-CNN'], loc='lower right')
# plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
plt.savefig("F1_for_all.pdf")  # should use before plt.show()
plt.show()
