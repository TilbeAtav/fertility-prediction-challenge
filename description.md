# Description of submission

In this submission I try to predict whether an individual in my sample will have another child in the years 2021-2023. 

I first clean the data by removing observations with  missing values in the outcome, retaining the variables of interest and then removing missing values in the features.

The features I use for predicting, and thus the ones I retain, are:

- Age in 2020;
- gender in 2020;
- intention to have another child in 2020;
- chance of loosing job in next 12 months in 2020.

All features are numeric and have varying means and standard dev. I therefore scale the variables using the StandardScaler. I split the data into a training (70%) and testing (30%) sample and fit the model using LogisticRegression after upscaling the minority sample in training data to account for unbalanced data (individuals that decide to have a child are a much smaller share of the sample). Finally, I fit the model on the test data and calculate the percision, recall and F1-score metrics for evaluation.

