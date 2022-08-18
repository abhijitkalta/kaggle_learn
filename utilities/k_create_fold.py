# Stratified k-fold for regression
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection

# Create k folds by stratifying on bins
def create_folds(data):
    # create a new column kfold and set all values to -1
    data['kfold'] = -1
    
    # Shuffle the rows of data
    data = data.sample(frac = 1).reset_index(drop = True)
    
    # Calculate the number of bins by sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    data.loc[:, 'bins'] = pd.cut(
        data['target'], bins = num_bins, labels = False
    )
    
    # Initialise the kfold class
    kf = model_selection.StratifiedKFold(n_splits = 5)
    
    # fill the kfold column
    for f, (t_, v_) in enumerate(kf.split(X = data, y = data.bins.values)):
        data.loc[v_, 'kfold'] = f
        
    # drop the bins column
    data = data.drop('bins', axis = 1)
    
    # Return data with folds
    return data


if __name__ == '__main__':

    # Create a sample data
    X, y = datasets.make_regression(
        n_samples = 15000, n_features = 100, n_targets = 1
    )
    
    # Create a dataframe out of the samples
    df = pd.DataFrame(
        X,
        columns = [f'f_{i}' for i in range(X.shape[1])]
    )
    df.loc[:, 'target'] = y
    
    # Create folds
    df = create_folds(df)