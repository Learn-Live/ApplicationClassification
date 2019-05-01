import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

svm = [0.30357142857142855, 0.42678571428571427, 0.59285714285714286, 0.69285714285714284, 0.71071428571428574,
       0.71250000000000002, 0.70714285714285718]
svm = np.asarray(svm, dtype=float) * 100

knn = [0.26964285714285713, 0.4642857142857143, 0.61250000000000004, 0.62857142857142856, 0.64464285714285718,
       0.63749999999999996, 0.62321428571428572]
knn = np.asarray(knn, dtype=float) * 100

GaussianNB = [0.30892857142857144, 0.34642857142857142, 0.5178571428571429, 0.55178571428571432, 0.42142857142857143,
              0.34107142857142858, 0.29821428571428571]
GaussianNB = np.asarray(GaussianNB, dtype=float) * 100

LR = [0.2767857142857143, 0.4732142857142857, 0.5357142857142857, 0.68571428571428572, 0.71785714285714286,
      0.72142857142857142, 0.72857142857142854]
LR = np.asarray(LR, dtype=float) * 100

CNN = [0.2875, 0.5589, 0.7893, 0.8679, 0.9161, 0.9107, 0.904]
CNN = np.asarray(CNN, dtype=float) * 100

DT = [0.51833887043189368, 0.7160132890365448, 0.8057142857142857, 0.85222591362126241, 0.87049833887043191,
      0.85056478405315615, 0.85554817275747506]
DT = np.asarray(DT, dtype=float) * 100

pseg = [50, 300, 800, 1500, 3000, 4000, 6000]

# l1=plt.plot(pseg,svm,'r--',label='type1')
# l2=plt.plot(pseg,knn,'g--',label='type2')
# l2=plt.plot(pseg,GaussianNB,'g--',label='type2')
plt.plot(pseg, svm, 'mx-', pseg, knn, 'g*-', pseg, GaussianNB, 'yp-', pseg, LR, 'ks-', pseg, DT, 'b^-', pseg, CNN,
         'ro-')
plt.ylabel("Classification accuracy (%)")
plt.xlabel('Input size (bytes) of classification models')
plt.ylim(0, 100)
# plt.tight_layout(.5)

plt.legend(['SVM', 'KNN', 'NB', 'LR', 'DT', '1D-CNN'], loc='lower right')
plt.savefig("traditionalML" + ".jpg", dpi=400)
plt.savefig("relation_inputsize_accuracy.pdf")  # should use before plt.show()

plt.show()
