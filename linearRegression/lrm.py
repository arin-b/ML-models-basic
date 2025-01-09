import copy, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)   # reduced display precision on numpy arrays


# The following lines create our training examples
# Here, our training set has 3 examples of house prices. X_train contains the 4 features each for the 3 houses
# y_train contains their final prices
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


# data is stored in numpy array/matrix as you can see below after execution
print("Our data is stored in a numpy array, as seen below:")
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)}")
print(y_train)
print("\n\n")


# for demonstration purposes, we have loaded parameter vectors w and b with some near-optimum values
# w is a 1-D numpy vector; b is a scalar
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"Parameter vector w (initial values): {w_init}")
print(f"Parameter vector b (initial value): {b_init}\n\n")


# we will now implement a function to make a prediction on a single training example from our X_train
# note that ours is a vectorized implementation
def predict(x, w, b):
    """
    prediction using linear regression
    Args:
    x: input example with multiple features (ndarray)
    w: parameter vector for above example (ndarray)
    b: parameter for above example (scalar)
    Returns:
    p: prediction (scalar)
    """
    p = np.dot(x, w) + b
    return p


# Note: we are using a vectorized version of the formula given below:
# f_wb(x) = w0x0 + w1x1 + w2x2 + ... + b

# Let us make a prediction for an example from our training set
x_example = X_train[0, :]  # getting a row from our training data
print(f"Our trial example is: {x_example}; its shape is: {x_example.shape}")

pred = predict(x_example, w_init, b_init)
print(f"Prediction: {pred}\n\n")
# notice that our prediction is very similar to the one present in y_train


def cost_function(X, y, w, b):
    """
    Compute the cost for all the predictions
    Args:
    X: input vector with all training examples (ndarray (m,n))
    y: vector with target values (ndarray (m,))
    w: parameter vector (ndarray (n, ))
    b: parameter value (scalar)
    Returns:
    cost: final cost (scalar)
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2
    cost = cost/(2*m)
    return cost

cost = cost_function(X_train, y_train, w_init, b_init)
print(f"The total cost at optimal paramters is: {cost}\n\n")


def calculate_gradients(X, y, w, b):
    """
    Computes gradient descent for linear regression
    Args:
    X: input vector (ndarray (m,n))
    y: target values (ndarray (m,))
    w: parameter vector (ndarray (n,))
    b: model parameter (scalar)
    Returns:
    dj_dw: gradient of cost w.r.t. the parameters w (ndarray (n,))
    dj_db: gradient of cost w.r.t. the parameter b (scalar)
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error*X[i, j]
        dj_db = dj_db + error
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db


temp_dj_dw, temp_dj_db =calculate_gradients(X_train, y_train, w_init, b_init)
print(f"Initial gradients for w: {temp_dj_dw}, \nInitial gradient for b: {temp_dj_db}\n\n")


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iterations):
    """
    Performs batch gradient descent to learn w and b
    Args:
    X: input vector (ndarray (m,n))
    y: target values (ndarray (m,))
    w_in: model params (ndarray (n,))
    b_in: model param (scalar)
    cost_function: function to compute cost
    gradient_function: function to compute gradients
    alpha: learning rate
    num_iterations: number of iterations to run gradient descent (scalar)
    """
    # an array to store J and w's at each iteration to be used for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # to avoid modifying global w inside function
    b = b_in

    for i in range(num_iterations):
        # calculate gradient and update parameters
        dj_dw, dj_db = gradient_function(X, y, w, b)

        # update parameters
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        # save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        # print cost at certain fixed intervals
        if i%math.ceil(num_iterations/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history


# now, we will initialize parameters to zero 
# below code runs complete linear regression for our training data
w_initial = np.zeros_like(w_init)
b_initial = 0.0

iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_history = gradient_descent(X_train, y_train, w_initial, b_initial, 
                                               cost_function, calculate_gradients,
                                               alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m, _ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")


# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_history)
ax2.plot(100 + np.arange(len(J_history[100:])), J_history[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()

# you will notice that our implementation is not ideal
# cost is still declining and predictions are not very accurate
# we need to yet perform feature scaling and hyperparameter tuning

