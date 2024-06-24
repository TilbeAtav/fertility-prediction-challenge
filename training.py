"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")
    
    features = [
        "cf20m003",
        "cf20m128", 
        "cf20m004",  
        "ci20m379",
        "cf20m013",
        "cf20m014",
        "cf20m015",
        "cf20m016",
        "cf20m020",
        "cf20m022",
        "cf20m024",
        "cf20m025",
        "cf20m027",
        "burgstat_2020",
        "woonvorm_2020",
        "oplmet_2020"
    ] 
    

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    
    # Upsampling only done for training sample, then applied to true test sample
    oversamp_df = resample(model_df[model_df["new_child"] == 1],
                        replace=True,
                        n_samples=model_df[model_df["new_child"] == 0].shape[0],
                        random_state=123)

    model_df = pd.concat((model_df[model_df["new_child"] == 0], oversamp_df))
    
    
    
    # Impute missing 
    imputer = KNNImputer(n_neighbors=2, weights="uniform").set_output(transform = "pandas")
    imputer.fit_transform(model_df[features])
    
    
    # Transform data
    numerical_columns = features
    #categorical_columns = [""]
    
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()
    
    preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])
    

    # Model
    model = make_pipeline(imputer, preprocessor, LogisticRegression())

    # Fit the model
    model.fit(model_df[features], model_df["new_child"])

    # Save the model
    joblib.dump(model, "model.joblib")