from numpy.random import permutation
from sklearn import svm, datasets
import neptune.new as neptune

def main():
    run = neptune.init(
    project="nexlson/iris-SVM",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMWI0YzlmZi03YzM2LTQxNTMtOTY2ZC0wZmE5NTNiOTRjZTQifQ==",
    )
    C = 1.0
    gamma = 0.7
    iris = datasets.load_iris()
    per = permutation(iris.target.size)
    iris.data = iris.data[per]
    iris.target = iris.target[per]
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(iris.data[:90],
          iris.target[:90])
    score = clf.score(iris.data[90:],
                   iris.target[90:])

    run["parameters"] = {'C': C, 'gamma': gamma}
    run['score'] = score

if __name__ == "__main__":
    main()