"""
.. module:: Partitional

Partitional
*************

:Description: Partitional

    

:Authors: bejar
    

:Version: 

:Created on: 18/09/2015 9:12 

"""

__author__ = 'bejar'


from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from pylab import *
import seaborn as sns
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score


iris = datasets.load_iris()

km = KMeans(n_clusters=3)

labels = km.fit_predict(iris['data'])

print(adjusted_mutual_info_score(iris['target'], labels))

plt.scatter(iris['data'][:, 2], iris['data'][:, 1], c=labels, s=100)
plt.show()


gmm = GMM(n_components=3, covariance_type='full')
gmm.fit(iris['data'])

labels = gmm.predict(iris['data'])
print(adjusted_mutual_info_score(iris['target'], labels))

plt.scatter(iris['data'][:, 2], iris['data'][:, 1], c=labels, s=100)
plt.show()
