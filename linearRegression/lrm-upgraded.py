import numpy as np
import matplotlib.pyplot as plt
import copy, math

# In the previous linear regression model lrm.py, we saw some unwanted results
# The model did not perform as well due to various reasons
# In this file, I will write a simple model but with some hyperparameter tuning to improve performance

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


