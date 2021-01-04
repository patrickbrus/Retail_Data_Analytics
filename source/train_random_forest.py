from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

from sklearn.ensemble import RandomForestRegressor

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "forest.joblib"))
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
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    
    # read in other hyperparameters
    n_estimators = args.n_estimators
    criterion = args.criterion
    max_depth = args.max_depth
    bootstrap = args.bootstrap

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    

    # Define a model 
    forest = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, bootstrap=bootstrap)
    
    # Train the model
    forest.fit(train_x, train_y)
    

    # Save the trained model
    joblib.dump(forest, os.path.join(args.model_dir, "forest.joblib"))