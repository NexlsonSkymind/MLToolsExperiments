from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment
from sacred.observers import MongoObserver

# Set experiment name here 
ex = Experiment('iris_rbf_svm')

# Add obsever (linked mongoDB)
ex.observers.append(MongoObserver(db_name="sacred"))

# Set configs here 
@ex.config
def cfg():
  C = 1.0
  gamma = 0.7

# Main here 
@ex.automain
def run(C, gamma):
  # load dataset
  iris = datasets.load_iris()
  per = permutation(iris.target.size)
  iris.data = iris.data[per]
  iris.target = iris.target[per]
  clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
  clf.fit(iris.data[:90],
          iris.target[:90])
  return clf.score(iris.data[90:],
                   iris.target[90:])