from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV

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
    
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=None)
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    
    # read in other hyperparameters
    n_iter = args.n_iter
    n_folds = args.n_folds
    n_jobs = args.n_jobs

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    

    # Define a model 
    forest = RandomForestRegressor(criterion="mse", n_jobs=n_jobs)
    
    # Create the Bayesion optimization object
    opt = BayesSearchCV(
        forest,
        {
            "max_depth": (5, 15),
            "n_estimators": (10, 50),
            "bootstrap": [True, False]
        },
        n_iter=n_iter,
        cv=n_folds
    )
    
    # Train the model
    opt.fit(train_x, train_y)
    
    model = opt.best_estimator_
    
    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))