import numpy as np
import random
from scipy.optimize import minimize
import time
from scipy import spatial as spt

"""
Python Sparse Variational Gaussian Process module.
This code is supposed to accompany the tutorial 'A Tutorial on
Sparse Variational Gaussian Processes'.

P.L.Green
University of Liverpool
21/05/19
"""


def Kernel(squared_distances, L):
    ''' Kernel function (squared exponential with length-scale L)

    '''

    return np.exp(-1/(2*L**2)*squared_distances)


def Train(L, Sigma, X, Y, N, M, NoCandidates):
    ''' Train Sparse GP

    '''

    X = np.vstack(X)    # Stack X vertically
    Y = np.vstack(Y)    # Stacy Y vertically
    start_time = time.time()    # Note the time when training began
    Theta = [L, Sigma]          # GP hyperparameters

    # Loop over all the candidate sparse points
    for r in range(NoCandidates):

        # Randomly select the locations of the candidate sparse points
        indices = random.sample(range(N), M)
        X_candidate = X[indices]

        # Define arguments to be passed to 'NegLowerBound' and
        # evaluate the lower bound for candidate sparse points
        a = (X, Y, X_candidate, N, M)
        LB = -NegLowerBound(Theta, a)

        if r == 0:              # For the first set of candidate points
            Xs = X_candidate    # Define optimum sparse inputs found so far
            Ys = Y[indices]     # Define optimum sparse outputs found so far
            LB_best = LB        # Store maximum lower bound found so far
        else:
            if LB > LB_best:      # If lower bound is largest we've seen
                Xs = X_candidate  # Define optimum sparse inputs found so far
                Ys = Y[indices]   # Define optimum sparse outputs found so far
                LB_best = LB      # Store maximum lower bound found so far

    # Update hyperparameters using scipy.minimize
    a = (X, Y, Xs, N, M)  # Arguments needed as input to 'minimize' function
    b1 = (1e-3, 3)        # Bounds on length scale
    b2 = (1e-3, 1)        # Bounds on noise standard deviation
    bnds = (b1, b2)

    # Search for optimum hyperparameters
    sol = minimize(NegLowerBound, x0=Theta, args=(a, ),
                   method='SLSQP', bounds=bnds)
    Theta = sol.x

    # Find final gram matrix (and its inverse)
    L, Sigma = Theta                         # Extract hyperparameters
    K = Kernel(FindSquareDistances(Xs), L)   # Gram matrix
    C = K + Sigma**2 * np.eye(M)             # C matrix
    invC = np.linalg.inv(C)                  # Find inverse of C
    elapsed_time = time.time() - start_time  # Time taken for training
    return L, Sigma, K, C, invC, Xs, Ys, LB_best, elapsed_time


def NegLowerBound(Theta, a):
    ''' Find negative lower bound (negative so we can use
        minimisation optimisation functions)

    '''

    X, Y, Xs, N, M = a  # Extract arguments
    L, Sigma = Theta    # Extract hyperparameters
    K_MM = Kernel(FindSquareDistances(Xs), L)     # Find K_MM
    InvK_MM = np.linalg.inv(K_MM)                 # Find inverse of K_MM
    K_NM = Kernel(FindSquareDistances(X, Xs), L)  # Find K_NM
    K_MN = np.transpose(K_NM)                     # Find K_MN

    # We define A = K_NM^T * invK_MM * K_MN
    A = np.dot(K_NM, np.dot(InvK_MM, K_MN))

    # B is an array containing only diagonal elements of K_NN - A.
    # Note we assume diagonal elements of A are always equal to 1.
    B = np.zeros(N)
    for i in range(N):
        B[i] = 1 - A[i, i]

    # Calculate the (negative) lower bound
    C = A+np.eye(N)*Sigma**2
    Sign, LogDetC = np.linalg.slogdet(C)
    LogDetC = Sign*LogDetC
    NLB = -(-0.5*LogDetC - 0.5*np.dot(Y.T, np.dot(np.linalg.inv(C), Y))
            - 1/(2*Sigma**2)*np.sum(B))
    return NLB.ravel()


def FindSquareDistances(X, Xm=None):
    ''' Finding the squared distances between sets of points.
        Note that this uses 'spatial' from scipy.

    '''

    if Xm is not None:
        return spt.distance.cdist(X, Xm, metric='sqeuclidean')
    else:
        return spt.distance.cdist(X, X, metric='sqeuclidean')


def Predict(X, xStar, L, Sigma, Y, K, C, InvC, N):
    ''' Standard GP prediction

    '''

    if np.size(X[0]) == 1:
        squared_distances = (X-xStar)**2
    else:
        squared_distances = np.sum((X-xStar)**2, 1)
    k = Kernel(squared_distances, L)
    c = 1 + Sigma**2   # Always true for this particular kernel
    yStarMean = np.dot(k.T, np.dot(InvC, Y))
    yStarStd = np.sqrt(c - np.dot(k.T, np.dot(InvC, k)))
    return yStarMean, yStarStd
