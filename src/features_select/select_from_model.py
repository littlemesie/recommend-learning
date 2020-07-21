from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston

def get_iris_data():
    iris = load_iris()
    X, y = iris.data, iris.target  # iris数据集
    return X, y

def get_boston_data():

    X, y = load_boston(return_X_y=True)
    label = []
    for i in y:
        if i >= 25:
            label.append(1)
        else:
            label.append(0)
    return X, label

def lr():
    """LogisticRegressionn 特征选择"""
    X, y = get_boston_data()
    lr_selector = SelectFromModel(LogisticRegression(max_iter=3000))
    lr_selector.fit(X, y)
    new_X = lr_selector.transform(X)
    print(new_X.shape)

def rm():
    """RandomForestClassifier 特征选择"""
    X, y = get_boston_data()
    rm_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold=0.05)
    rm_selector.fit(X, y)

    new_X = rm_selector.transform(X)
    print(new_X)

rm()
