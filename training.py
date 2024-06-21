"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample



def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    random.seed(1) # not useful here because logistic regression deterministic
    
    data_oversampled, target_oversampled = resample(df[outcome_df == 1], 
                                           outcome_df[outcome_df == 1],
                                           replace=True,
                                           n_samples= df[outcome_df == 0].shape[0],
                                           random_state=123)

    df = pd.concat((df[outcome_df == 0], data_oversampled))
    outcome_df = pd.concat((df[outcome_df == 0], target_oversampled))

    # Combine cleaned_df and outcome_df
    model_df = pd.merge(df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    
    # Logistic regression model
    model = LogisticRegression(max_iter=500)

    # Fit the model
    model.fit(model_df[['age']], model_df['new_child'])

    # Save the model
    joblib.dump(model, "model.joblib")
