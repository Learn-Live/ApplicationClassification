from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
y_true = [0, 0, 0, 0, 0, 0]
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(precision_score(y_true, y_pred, average='macro'))
# 0.22...
print(precision_score(y_true, y_pred, average='micro'))
# 0.33...
print(precision_score(y_true, y_pred, average='weighted'))
# 0.22...
print(precision_score(y_true, y_pred, average=None))
# array([0.66..., 0.        , 0.        ])

print()

print(recall_score(y_true, y_pred, average='macro'))
# 0.22...
print(recall_score(y_true, y_pred, average='micro'))
# 0.33...
print(recall_score(y_true, y_pred, average='weighted'))
# 0.22...
print(recall_score(y_true, y_pred, average=None))
# array([0.66..., 0.        , 0.        ])


y_pred = [0, 0, 0, 0, 0, 0]
precision_score(y_true, y_pred, average=None)
# array([0.33..., 0.        , 0.        ])
precision_score(y_true, y_pred, average=None, zero_division=1)
# array([0.33..., 1.        , 1.        ])
