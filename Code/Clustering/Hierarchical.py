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

from sklearn import datasets
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pylab import *
import seaborn as sns
import numpy as np


iris = datasets.load_iris()

# mdist = pairwise_distances(iris['data'])
# md = np.tril(mdist)
clust = linkage(iris['data'], method='single')
# print clust
# dendrogram(clust)
# plt.show()

labels = fcluster(clust, 3, criterion='maxclust')

print labels
plt.figure(figsize=(10,10))
plt.scatter(iris['data'][:, 2], iris['data'][:, 1], c=labels,s=100)
plt.show()