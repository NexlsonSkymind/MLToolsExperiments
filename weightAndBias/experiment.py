import wandb
from numpy.random import permutation
from sklearn import svm, datasets

def main():
    # Initialize WandB 
    wandb.init(name='Iris SVM', 
            project='model Experiments',
            notes='This is a test run', 
            tags=['Fashion MNIST', 'Test Run'],
            entity='Nexlson')

    iris = datasets.load_iris()
    per = permutation(iris.target.size)
    iris.data = iris.data[per]
    iris.target = iris.target[per]
    C = 1.0
    gamma = 0.7
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(iris.data[:90],
          iris.target[:90])
    score = clf.score(iris.data[90:],
                   iris.target[90:])
    # set configs
    wandb.config.C = C
    wandb.config.gamma = gamma
    
    # set metrics
    wandb.log({'score': score})


if __name__ == "__main__":
    main()