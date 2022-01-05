from comet_ml import Experiment 
from numpy.random import permutation
from sklearn import svm, datasets

def main():
    # Initiate 
    experiment = Experiment(api_key="fganx55aE3umFmx5iBWiuckxc", project_name="Iris SVM", workspace="nexlson")

    C = 1.0
    gamma = 0.8
    iris = datasets.load_iris()
    per = permutation(iris.target.size)
    iris.data = iris.data[per]
    iris.target = iris.target[per]
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(iris.data[:90],
          iris.target[:90])
    score = clf.score(iris.data[90:],
                   iris.target[90:])
    
    experiment.log_parameters({'C': C, 'gamma': gamma})
    experiment.log_metrics({'score': score})

if __name__ == "__main__":
    main()