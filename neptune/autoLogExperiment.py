from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
from keys import NEPTUNE_API

# Neptune setup
run = neptune.init(
    project="nexlson/sklearn-integration",
    api_token=NEPTUNE_API,
) 

# hyperparams
parameters = {'n_estimators': 50,
              'max_depth': 8,
              'min_samples_split': 2}

# model
rfr = RandomForestRegressor(**parameters)

# data preps
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

# model trainning 
rfr.fit(X_train, y_train)

# data logging
run['rfr_summary'] = npt_utils.create_regressor_summary(rfr, X_train, X_test, y_train, y_test)