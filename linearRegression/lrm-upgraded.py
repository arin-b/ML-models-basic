import numpy as np
import matplotlib.pyplot as plt
import copy, math
import pandas as pd

# In the previous linear regression model lrm.py, we saw some unwanted results
# The model did not perform as well due to various reasons
# In this file, I will write a simple model but with some hyperparameter tuning to improve performance

# In the below lines, we import our data
# The data was taken from:
# https://www.kaggle.com/datasets/anmolkumar/house-price-prediction-challenge?select=test.csv

# First, load the data into a dataframe
df = pd.read_csv("train.csv")
# We will now remove some columns to simplify our data
df = df.drop(['POSTED_BY', 'BHK_OR_RK', 'ADDRESS'], axis=1)
# Copy the target value into a separate dataframe, then remove it from original
target_df = df[['TARGET(PRICE_IN_LACS)']].copy()
target_df.reset_index(drop=True, inplace=True)
# Convert our dataframes into numpy arrays
x_train = df.to_numpy()
y_train = target_df.to_numpy()
# Our data preprocessing ends here
# Now, we start creating the required functions for our model

# Randomly initialize model parameters
w_init = 2
b_init = 1


# Write function to compute cost
def compute_cost(x, y, w, b):
    """
    x: input data (ndarray)
    y: target values (ndarray)
    w: model parameters (ndarray)
    b: model parameter (scalar)
    Returns:
    total_cost: total cost calculated for the dataset (float)
    """
    m = x.shape[0]

    total_cost = 0
    f = 0
    for i in range(m):
        f = (w*x[i])+b
        total_cost += (f-y[i])**2
    total_cost = total_cost/(2*m)

    return total_cost


# Testing whether the function works
cost = compute_cost(x_train, y_train, w_init, b_init)
print(f'Cost at initial w: {cost}')



