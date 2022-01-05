from numpy.random import permutation
from sklearn import svm, datasets
import modelstash 
import joblib 

def main():
    # Instantiate modelstash
    modelstash = modelstash.ModelStash(url="http://localhost:8080/", username="user", password="asht")

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
    # save model
    joblib.dump(clf, "irisSVM.pkl")

    metric = {'score': score}
    config = {'C': C, 'gamma': gamma}

    # create model 
    modelstash.create_model(file_path="./irisSVM.pkl", model_name="Iris SVM", created_by="nelson", training_dataset="Iris dataset",  
      input_names=["Iris data"], output_names=["Prediction"], training_metric=metric, model_config=config)

if __name__ == "__main__":
    main()