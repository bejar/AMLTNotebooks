"""
.. module:: Distances

Distances
*************

:Description: Distances

    

:Authors: bejar
    

:Version: 

:Created on: 08/09/2015 11:53 

"""

__author__ = 'bejar'


from sklearn import datasets
from pylab import *
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

iris = datasets.load_iris()
col = ['r', 'g', 'b']
lc = [col[i] for i in iris['target']]

mdist = pairwise_distances(iris['data'], metric='euclidean')
fig = plt.figure(figsize=(8,8))
fig.add_subplot(121)
plt.imshow(mdist)
#plt.show()

mds = MDS(n_components=2,dissimilarity='precomputed', random_state=0)
fdata = mds.fit_transform(mdist)


fig.add_subplot(122)
plt.scatter(fdata[:, 0], fdata[:, 1], c=lc,s=100)
plt.show()



mdist = pairwise_distances(iris['data'], metric='cosine')

plt.imshow(mdist)
plt.show()

fdata = mds.fit_transform(mdist)
fig = plt.figure(figsize=(8,8))
plt.scatter(fdata[:, 0], fdata[:, 1], c=lc,s=100)
plt.show()

mdist = pairwise_distances(iris['data'], metric='cityblock')
plt.imshow(mdist)
plt.show()

fdata = mds.fit_transform(mdist)
fig = plt.figure(figsize=(8,8))
plt.scatter(fdata[:, 0], fdata[:, 1], c=lc,s=100)
plt.show()

mdist = pairwise_distances(iris['data'], metric='mahalanobis')
plt.imshow(mdist)
plt.show()

fdata = mds.fit_transform(mdist)
fig = plt.figure(figsize=(8,8))
plt.scatter(fdata[:, 0], fdata[:, 1], c=lc,s=100)
plt.show()
