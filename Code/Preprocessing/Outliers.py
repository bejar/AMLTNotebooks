"""
.. module:: Outliers

Outliers
*************

:Description: Outliers

    

:Authors: bejar
    

:Version: 

:Created on: 27/11/2017 8:25 

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sn

__author__ = 'bejar'


rng = np.random.RandomState(42)

# Example settings
n_samples = 200
outliers_fraction = 0.2


lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=outliers_fraction)



np.random.seed(42)
# Data generation
mean1 = [0, 0]
mean2 = [3.5, 4]
cov1 = [[1.5, -0.3], [-0.2, .5]]
cov2 = [[0.75, 0.4], [0.3, 0.5]]



X = np.r_[np.random.multivariate_normal(mean1, cov1, 100), np.random.multivariate_normal(mean2, cov2, 100)]
# Add outliers

y_pred = lof.fit_predict(X)
scores_pred = lof.negative_outlier_factor_


plt.figure(figsize=(18, 9))


subplot = plt.subplot(1, 2, 1)
b = subplot.scatter(X[:, 0], X[:, 1], c=['k' if y == 1 else 'r' for y in y_pred], s=20)
subplot = plt.subplot(1, 2, 2)
b = subplot.scatter(X[:, 0], X[:, 1], c=-np.log(-scores_pred), s=20, cmap=plt.get_cmap('Reds'))

subplot.axis('tight')


plt.show()