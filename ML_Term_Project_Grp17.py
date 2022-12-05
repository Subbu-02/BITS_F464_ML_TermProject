# Importing the necassary modules
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.decomposition import SparsePCA
from sklearn.datasets import load_digits
from sklearn.linear_model import ElasticNet
import numpy.linalg as linalg

# Using the make_friedman1 function to generate a random dataset
X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
transformer = SparsePCA(n_components=5, random_state=0)
transformer.fit(X)
X_transformed = transformer.transform(X)

# Using the load_digits function to generate a random dataset
digits = load_digits()
sparse_pca = SparsePCA(n_components=60, alpha=0.1)
sparse_pca.fit_transform(digits.data / 255)

# Implementation of the Elastic Net in function spca
def spca(n_components, lambdas, ridge_alpha, max_iter, lim, data):
  '''
  n_components indicates the number of principal components to be extracted
  lambdas includes the L1 penalties
  ridge_alpha is the penalty of ridge regression
  max_iter indicates the maximum number of iterations
  lim indicates the required limit of convergence
  data is the matrix on which we perform spca
  '''
  U, D, V = linalg.svd(data) #extracting components on performing SVD
  A_orig = np.transpose(data[:n_components]) #intialize A
  B_orig = np.zeros(A_orig.shape()) #initialize B
  B = np.zeros(A_orig.shape())
  mat = (np.transpose(V) * D) @ V #set matrix to be fit by the elastic net

  for i in range(max_iter):
    # for fixed A
    for i in range(n_components):
      model = ElasticNet(alpha = (2 * ridge_alpha + lambdas[i]) / (2 * n), l1_ratio = lambdas[i]/(2*ridge_alpha + lambdas[i])) #initialize elastic net
      model.fit(mat, mat @ A[:, i]) #fit to the required (y-X)
      B[:, i] = model.coef_ #update B

    # for fixed B
    U, D, V = linalg.svd((np.transpose(data) @ data) @ B) #perform SVD
    A = U @ V #update A as UV transpose

    if (linalg.norm(A - A_orig) < lim) and (linalg.norm(B - B_orig) < lim): #check for convergence
      break
    
    A_orig = A #update A_orig
    B_orig = B #update B_orig

  B = B / linalg.norm(B, 2, axis=0)  #normalization
  return A, B
