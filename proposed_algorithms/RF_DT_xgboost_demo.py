from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from numpy_load_and_arff import load_npy_data
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def main_xgboost(X, Y, session_size=2000, test_percent=0.1):
    input_size = 500
    reduce_feature_flg = False
    if reduce_feature_flg:
        print(f'Using PCA to reduce features.')
        sc = StandardScaler()
        X_train = sc.fit_transform(X)
        X = sc.transform(X)
        pca_model = PCA(n_components=input_size, random_state=0)
        pca_model.fit_transform(X, y)
        X = pca_model.transform(X)
        explained_variance = pca_model.explained_variance_ratio_
        print(f'explained_variance={explained_variance}')

    # session_size = input_size  # X.shape[1]
    print(f'X.shape={X.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=42)
    print(f'train_test_ratio:[{1-test_percent}:{test_percent}]')
    # listdir = {}
    # for i in range(0, len(y_train)):
    #     if y_train[i] not in listdir:
    #         listdir.update({y_train[i]: 0})
    #     else:
    #         listdir[y_train[i]] = listdir[y_train[i]] + 1
    # print(f'X_train:{listdir}')
    #
    # listdir = {}
    # for i in range(0, len(y_test)):
    #     if y_test[i] not in listdir:
    #         listdir.update({y_test[i]: 0})
    #     else:
    #         listdir[y_test[i]] = listdir[y_test[i]] + 1
    # print(f'X_test:{listdir}')

    # train&test
    result = []
    # for i in range(10,300,30):
    #    value.append(i)
    # value = [100]
    value = 100
    print("n_estimators: ", value)
    # truncatelist = [10, 100, 300, 500, 3000, 6000, session_size]
    truncatelist = [i * 100 + 500 for i in range(50, 100, 5)]
    print(f'{truncatelist}')
    # for i in range(50,1500,50):
    for i in truncatelist:
        print(f'session_size:{i}')
        # clf = RandomForestClassifier(n_estimators=value, min_samples_leaf=2)
        # clf = RandomForestClassifier()
        # clf = DecisionTreeClassifier(criterion="entropy", splitter="best",
        #                              max_depth = 20,
        #                              # min_samples_split=5,
        #                              min_samples_leaf=5,
        #                              # min_weight_fraction_leaf=0.,
        #                              # max_features=None,
        #                              random_state=20)
        clf = DecisionTreeClassifier(random_state=20)
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
        # print(result)
        print(f'test acc: {result}')


if __name__ == '__main__':
    input_file = '../input_data/trdata-8000B_payload.npy'
    # input_file = '../input_data/trdata-8000B_header_payload_20190326.npy'
    input_file = '../input_data/trdata_P_8000.npy'   # test acc: [0.26696329254727474, 0.42936596218020023, 0.43492769744160176, 0.492769744160178, 0.6651835372636262, 0.6685205784204672]
    # input_file = '../input_data/trdata_PH_8000.npy'  # test acc: [0.22024471635150167, 0.5183537263626251, 0.5717463848720801, 0.610678531701891, 0.8153503893214683, 0.8209121245828699]
    # input_file = '../input_data/trdata_PHT_8000.npy' # test acc: [0.27697441601779754, 0.5116796440489433, 0.6028921023359288, 0.6062291434927698, 0.8075639599555061, 0.8186874304783093]
    # input_file = '../input_data/trdata_PT_8000.npy'  # test acc: [0.24916573971078976, 0.45161290322580644, 0.5617352614015573, 0.5761957730812013, 0.7552836484983315, 0.7552836484983315]
    # input_file ='../input_data/trdata_PT_8000_padding.npy'  # test acc: [0.389321468298109, 0.5828698553948832, 0.6262513904338154, 0.6551724137931034, 0.8731924360400445, 0.8921023359288098]
    input_file = '../input_data/newapp_10220_pt.npy'
    session_size = 6000
    X, y = load_npy_data(input_file, session_size)
    main_xgboost(X, y, session_size)
