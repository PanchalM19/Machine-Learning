import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

np.random.seed(7)
N = 10000

mu0 = np.array([-1, 1,-1, 1])
mu1 = np.array([1,1,1,1])

sig0 = np.array([[2, -0.5, 0.3, 0],[-0.5, 1, -0.5, 0],[0.3, -0.5, 1, 0], [0, 0, 0, 2]])
sig1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])

pri0 = 0.7
pri1 = 0.3

X = np.zeros(4,N)
labels = np.zeros(1,N)

for i = range(1,N)
    if rand < PY0
% class 0
X(:, i) = mvnrnd(mu0, Sigma0);
labels(i) = 0;
else
% class 1
X(:, i) = mvnrnd(mu1, Sigma1);
labels(i) = 1;
end
end

pdf0 = np.random.multivariate_normal(mu0, sig0, N)
# print(pdf0)
pdf1 = np.random.multivariate_normal(mu1, sig1, N)
lr = pdf1/pdf0

