from mlflow import log_metric, log_param, log_artifacts
from numpy.random import permutation
from sklearn import svm, datasets

def main():
    # hyperparams
    C = 1.0
    gamma = 0.7

    # dataset preps
    iris = datasets.load_iris()
    per = permutation(iris.target.size)
    iris.data = iris.data[per]
    iris.target = iris.target[per]

    # Model
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(iris.data[:90],
          iris.target[:90])
    score = clf.score(iris.data[90:],
                   iris.target[90:])
    
    # MLFLow tracking logs data
    log_param("C", C)
    log_param("gamma", gamma)
    log_metric("score", score)
    log_artifacts("./results")

if __name__ == "__main__":
    main()