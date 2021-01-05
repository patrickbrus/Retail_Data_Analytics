from __future__ import print_function

import argparse
import os
import pandas as pd
import numpy as np

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model

if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument('--criterion', type=str, default="mse")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--bootstrap", type=bool, default=True)
    
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=3)
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    
    # read in other hyperparameters
    n_iter = args.n_iter
    n_folds = args.n_folds

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    hyperparameters = {
        'learning_rate': (0.01, 1.0),
        'n_estimators': (100, 1000),
        'max_depth': (3,10),
        'subsample': (1.0, 1.0),  # Change for big datasets
        'colsample': (1.0, 1.0),  # Change for datasets with lots of features
        'gamma': (0, 5)
    }
    
    # function for creating and training a XGBoost classifier
    # return: mse
    def xgboost_hyper_param(learning_rate, n_estimators, max_depth, subsample, colsample, gamma):
        max_depth = int(max_depth)
        n_estimators = int(n_estimators)
        clf = XGBRegressor(
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            gamma=gamma)
        return np.mean(cross_val_score(clf, train_x, train_y, cv=n_folds, scoring='mse'))
    
    optimizer = BayesianOptimization(f=xgboost_hyper_param, pbounds=hyperparameters)
    
    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))