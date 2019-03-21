from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn_issues.numpy_load import load_npy_data


def main_xgboost(X, Y, session_size=2000, test_percent=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=42)
    print(f'train_test_ratio:[{1-test_percent}:{test_percent}]')
    listdir = {}
    for i in range(0, len(y_train)):
        if y_train[i] not in listdir:
            listdir.update({y_train[i]: 0})
        else:
            listdir[y_train[i]] = listdir[y_train[i]] + 1
    print(f'X_train:{listdir}')

    listdir = {}
    for i in range(0, len(y_test)):
        if y_test[i] not in listdir:
            listdir.update({y_test[i]: 0})
        else:
            listdir[y_test[i]] = listdir[y_test[i]] + 1
    print(f'X_test:{listdir}')

    # train&test
    result = []
    # for i in range(10,300,30):
    #    value.append(i)
    # value = [100]
    value = 5000
    print("n_estimators: ", value)
    truncatelist = [session_size]
    # for i in range(50,1500,50):
    for i in truncatelist:
        print(f'session_size:{session_size}')
        # clf = RandomForestClassifier(n_estimators=value, min_samples_leaf=2)
        clf = DecisionTreeClassifier(criterion="entropy", splitter="best",
                                     max_depth = 20,
                                     # min_samples_split=5,
                                     min_samples_leaf=5,
                                     # min_weight_fraction_leaf=0.,
                                     # max_features=None,
                                     random_state=20)
        # clf = XGBClassifier(n_estimators=150)
        X_train_t = X_train[:, :i]

        X_test_t = X_test[:, :i]
        print("before input....")
        print(f'X_train_t.shape:{X_train_t.shape}')
        print(y_train.shape)
        print(f'X_test_t.shape:{X_test_t.shape}')
        print(y_test.shape)
        # print((X_train_t[0])[0:10])
        clf.fit(X_train_t, y_train)

        predtrain = clf.predict(X_train_t)
        print(confusion_matrix(y_train, predtrain))

        predtest = clf.predict(X_test_t)
        print(confusion_matrix(y_test, predtest))

        print("train acc:", metrics.accuracy_score(y_train, predtrain))
        print("test acc", metrics.accuracy_score(y_test, predtest))

        result.append(metrics.accuracy_score(y_test, predtest))
        print(result)
    print(result)


if __name__ == '__main__':
    input_file = '../input_data/trdata-8000B.npy'
    session_size = 8000
    X, y = load_npy_data(input_file, session_size)
    main_xgboost(X, y, session_size)
