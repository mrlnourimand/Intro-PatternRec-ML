"""
DATA.ML.100: Introduction to Pattern Recognition and Machine Learning
Ex 06, title: Disease - Scikit-Learn regressors

(This program uses training data about measuring severity of a certain disease
(the larger is the value the worst is the case) and tries to classify the test
data with 4 different methods: the training data mean value(baseline), linear
model, Decision tree regressor, and Random forest regressor. For each of them
prints MSE.

Creator: Maral Nourimand
Student id number: 151749113
Email: maral.nourimand@tuni.fi
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn import ensemble


def main():
    # Load the training and test data
    x_train = np.loadtxt('disease_X_train.txt')
    x_test = np.loadtxt('disease_X_test.txt')
    y_train = np.loadtxt('disease_y_train.txt')  # training labels
    y_test = np.loadtxt('disease_y_test.txt')  # test labels

    # Calculate the mean of the training labels
    mean_train = np.mean(y_train)

    # Create an array of the same shape as y_test with the mean value
    baseline_predictions = np.full_like(y_test, mean_train)

    # Calculate the Mean Squared Error (MSE) for the baseline predictions
    mse_baseline = mean_squared_error(y_test, baseline_predictions)

    # Print the baseline MSE
    print(f"Baseline MSE: {mse_baseline:.3f} \n")

    # Fit a Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)

    # Use the linear model to predict test data
    linear_predictions = linear_model.predict(x_test)

    # Calculate the Mean Squared Error (MSE) for the linear model predictions
    mse_linear = mean_squared_error(y_test, linear_predictions)

    # Print the linear model MSE
    print(f"Linear Model MSE: {mse_linear:.3f} \n")

    # Fit a Decision Tree Regressor model
    # decision_tree_model = DecisionTreeRegressor()
    decision_tree_model = tree.DecisionTreeRegressor()
    decision_tree_model.fit(x_train, y_train)

    # Use the decision tree model to predict test data
    decision_tree_predictions = decision_tree_model.predict(x_test)

    # Calculate the (MSE) for the decision tree model predictions
    mse_decision_tree = mean_squared_error(y_test, decision_tree_predictions)

    # Print the decision tree model MSE
    print(f"Decision Tree Regressor MSE: {mse_decision_tree:.3f} \n")

    # Fit a Random Forest Regressor model
    # random_forest_model = RandomForestRegressor()
    random_forest_model = ensemble.RandomForestRegressor()
    random_forest_model.fit(x_train, y_train)

    # Use the random forest model to predict test data
    random_forest_predictions = random_forest_model.predict(x_test)

    # Calculate the (MSE) for the random forest model predictions
    mse_random_forest = mean_squared_error(y_test, random_forest_predictions)

    # Print the random forest model MSE
    print(f"Random Forest Regressor MSE: {mse_random_forest:.3f}")


if __name__ == "__main__":
    main()
