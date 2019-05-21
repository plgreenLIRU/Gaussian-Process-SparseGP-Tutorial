import numpy as np
import matplotlib.pyplot as plt
import pickle
import LIRU_SparseGP as sGP
"""
This code runs a 1D GP regression example using the module LIRU_SparseGP
P.L.Green
University of Liverpool
09/09/18
"""


# Make some 1D training data
N = 500                         # 500 training poitns
X = np.linspace(0, 10, N)       # Inputs evenly spaced between 0 and 10
F = np.sin(X)                   # True function (f = sin(x))
Y = F + 0.1*np.random.randn(N)  # Observations

# Initial hyperparameters
L0 = 0.5        # Lengthscale
Sigma0 = 0.2    # Noise standard deviation

# Train sparse GP
M = 10                  # No. sparse points
NoCandidates = 100      # No. of candidate sets of sparse points analysed
(L, Sigma, K, C, InvC, Xs, Ys, LB_best, elapsed_time) = sGP.Train(L0, Sigma0,
                                                                  X, Y, N, M,
                                                                  NoCandidates)

# Print results
print('Maximum lower bound:', LB_best)
print('Hyperparameters:', L, Sigma)
print('Elapsed time:', elapsed_time)

# Make some predictions
X_Star = np.linspace(0, 10, 200)  # Make predictions
N_Star = len(X_Star)              # No. points where we make predictions
Y_StarMean = np.zeros(N_Star)     # Initialise GP mean predictions
Y_StarStd = np.zeros(N_Star)      # Initialise GP std predictions
for n in range(N_Star):
    xStar = X_Star[n]
    Y_StarMean[n], Y_StarStd[n] = sGP.Predict(Xs, xStar, L, Sigma,
                                              Ys, K, C, InvC, M)

# Plot results
plt.figure()
plt.plot(X_Star, Y_StarMean, 'black', label='Sparse GP')
plt.plot(X_Star, Y_StarMean + 3 * Y_StarStd, 'black')
plt.plot(X_Star, Y_StarMean - 3 * Y_StarStd, 'black')
plt.plot(X, Y, '.', color='blue', label='Full dataset')
plt.plot(Xs, Ys, 'o', markeredgecolor='black', markerfacecolor='red',
         markeredgewidth=1.5, markersize=10, label='Sparse dataset')
plt.xlabel('x')
plt.legend()
plt.show()
