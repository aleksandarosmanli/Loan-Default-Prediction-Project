# Loan-Default-Prediction-Project

Project for predicting client's financial loan default. The financial institution has asked me to build a machine-learning model to predict if its client will fail to pay his loan to the financial institution. I was provided with a dataset containing a sample of loans for the period of one year.

*Dataset Description*
The provided dataset has one row of data for each client and it is divided into train and test sets. The training set contains 70% of the overall sample (255,347 clients and importantly, will reveal whether or not the client pay his loan (the “ground truth”). The testing dataset contains the same information about the remaining segment of the overall sample (109,435 clients) but does not disclose the “ground truth” for each client. The goal of this project is to predict this outcome. Both train and test datasets contain one row for each unique client. For each client, a single observation (LoanID) is included. In addition to this identifier column, the training dataset also contains the task's target label, a binary column 'Default'. In addition to that column, both datasets have an identical set of features that can be used to train the model to make predictions.

*Data Loading and Preparation*
I imported the necessary Python modules and I loaded the datasets. After that, I explored, cleaned, and validated the data. The datasets contain 2 features with float, 8 features with integer (including 'Default' feature in the training set), and 8 features with object type. 2 float and integer features are continuous, and 7 object features are categorical (the 8th object feature is 'LoanID'). Taking into account this dataset structure, I chose tree-based models, especially CatBoost which can effectively use the categorical features.

*Feature Engineering*
During this phase, I engineered new interaction features among the continuous features and their interaction with loan default. Next, I checked the correlations between loan default and all the categorical features' values to find some correlation with user churn.
 
By using the correlation matrix and Pearson correlation method, I plotted a correlation heatmap.
Next, I filtered the correlation between all the features (both basic and combined) with the loan default target variable. From the heatmap, I identified 4 interaction features highly correlated with other features. Taking into account that tree-based models are resilient to multicollinearity, I left all the 44 features in order to retain the predictive power to loan default from every feature.

*Numerical Features Scaling*
Because of very different ranges of numerical features' values, it is recommended to scale the numerical features before training the chosen models.

*Assigning the predictors and the target variable*
I split the training set into training and validation sets in order to check the performance of the model on the validation set before the prediction on the test dataset.

*Testing with different tree-based ML models*
I conducted tests on three different tree-based ML models:

LightGBM,
XGBoost,
CatBoost, and
HistGradientBoost
Also, Soft Voting Ensemble and Stacking Ensemble Model of LightGBM, XGBoost, CatBoost and Gradient Boosting Models were used to combine the predictive power of the individual models.
