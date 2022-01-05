from comet_ml import Experiment 
from numpy.random import permutation
from sklearn import svm, datasets
from keys import COMET_API

def main():
    # Comet Setup
    experiment = Experiment(api_key=COMET_API, project_name="Iris SVM", workspace="nexlson")

    # Hyperparams 
    C = 0.8
    gamma = 1.0

    # Dataset preps
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
    
    # Data logs
    experiment.log_parameters({'C': C, 'gamma': gamma})
    experiment.log_metrics({'score': score})

if __name__ == "__main__":
    main()