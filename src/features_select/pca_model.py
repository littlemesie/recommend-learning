from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def pca():
    iris = load_iris()
    X, y = iris.data, iris.target  # iris数据集
    model = PCA(n_components=2)
    model.fit(X)
    X_new = model.transform(X)
    print(X_new)

pca()