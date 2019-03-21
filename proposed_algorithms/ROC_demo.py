#   Note: this implementation is restricted to the binary classification task.

import numpy as np
from sklearn import metrics
y = np.array([1, 1, 0, 0])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores)
print(fpr, tpr, thresholds)
# array([ 0. ,  0.5,  0.5,  1. ])

