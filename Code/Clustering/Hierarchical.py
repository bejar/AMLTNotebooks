"""
.. module:: Hierarchical

Hierarchical
*************

:Description: Hierarchical

    

:Authors: bejar
    

:Version: 

:Created on: 08/09/2015 7:41 

"""

__author__ = 'bejar'

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pylab import *
import seaborn as sns
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

iris = datasets.load_iris()

# Single Linkage

clust = linkage(iris['data'], method='single')
fig = plt.figure(figsize=(16,8))
fig.add_subplot(121)
dendrogram(clust)
labels = fcluster(clust, 3, criterion='maxclust')
print(adjusted_mutual_info_score(iris['target'], labels))
fig.add_subplot(122)
plt.scatter(iris['data'][:, 2], iris['data'][:, 1], c=labels,s=100)
plt.show()

# Complete Linkage

clust = linkage(iris['data'], method='complete')
fig = plt.figure(figsize=(16,8))
fig.add_subplot(121)
dendrogram(clust)
labels = fcluster(clust, 3, criterion='maxclust')
print(adjusted_mutual_info_score(iris['target'], labels))
fig.add_subplot(122)
plt.scatter(iris['data'][:, 2], iris['data'][:, 1], c=labels,s=100)
plt.show()

# Average criterion

clust = linkage(iris['data'], method='average')
fig = plt.figure(figsize=(16,8))
fig.add_subplot(121)
dendrogram(clust)
labels = fcluster(clust, 3, criterion='maxclust')
print(adjusted_mutual_info_score(iris['target'], labels))
fig.add_subplot(122)
plt.scatter(iris['data'][:, 2], iris['data'][:, 1], c=labels,s=100)
plt.show()

# Ward criterion

clust = linkage(iris['data'], method='ward')
fig = plt.figure(figsize=(16,8))
fig.add_subplot(121)
dendrogram(clust)
labels = fcluster(clust, 3, criterion='maxclust')
print(adjusted_mutual_info_score(iris['target'], labels))
fig.add_subplot(122)
plt.scatter(iris['data'][:, 2], iris['data'][:, 1], c=labels,s=100)
plt.show()



