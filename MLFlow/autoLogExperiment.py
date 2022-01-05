import mlflow 
from numpy.random import permutation
from sklearn import svm, datasets

def main():
    # enable autologging
    mlflow.sklearn.autolog()

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

    # MLflow start train 
    with mlflow.start_run() as run:    
        clf.fit(iris.data[:90], iris.target[:90])
        print("Logged data and model in run {}".format(run.info.run_id))

if __name__ == "__main__":
    main()