from numpy.random import permutation
from sklearn import svm, datasets
import neptune.new as neptune
from keys import NEPTUNE_API

def main():
    # Neptune setup
    run = neptune.init(
    project="nexlson/iris-SVM",
    api_token=NEPTUNE_API)
    
    # Hyperparams 
    C = 1.0
    gamma = 0.7

    # Dataset Preps
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

    # Log data
    run["parameters"] = {'C': C, 'gamma': gamma}
    run['score'] = score

if __name__ == "__main__":
    main()