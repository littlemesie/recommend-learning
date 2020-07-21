from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def get_data():
    iris = load_iris()
    X, y = iris.data, iris.target  # iris数据集
    return X, y

def lr():
    """LogisticRegressionn 特征选择"""
    X, y = get_data()
    lr_selector = SelectFromModel(LogisticRegression())
    lr_selector.fit(X, y)
    new_X = lr_selector.transform(X)
    print(new_X)

def rm():
    """RandomForestClassifier 特征选择"""
    X, y = get_data()
    rm_selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
    rm_selector.fit(X, y)

    new_X = rm_selector.transform(X)
    print(new_X)

rm()
