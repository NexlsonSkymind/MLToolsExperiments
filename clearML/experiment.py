from clearml import Task
from numpy.random import permutation
from sklearn import svm, datasets

# ClearML setup
task = Task.init(project_name="Test project", task_name="best experiment 2")

# Hyperparams 
C = 0.6
gamma = 1.2

# Dataset Preps
iris = datasets.load_iris()
per = permutation(iris.target.size)
iris.data = iris.data[per]
iris.target = iris.target[per]

# Model 
clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
clf.fit(iris.data[:90],
        iris.target[:90])
clf.score(iris.data[90:],
                iris.target[90:])

