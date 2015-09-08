"""
.. module:: DimReductionArt

DimReductionArt
*************

:Description: DimReductionArt

    

:Authors: bejar
    

:Version: 

:Created on: 07/09/2015 15:38 

"""

__author__ = 'bejar'

from sklearn.datasets import load_digits
from pylab import *
#import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from amltlearn.datasets import make_blobs


blobs, labels = make_blobs(n_samples=200, n_features=3, centers=[[1,1,1],[0,0,0],[-1,-1,-1]], cluster_std=[0.2,0.1,0.3])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.scatter(blobs[:, 0], blobs[:, 1], zs=blobs[:, 2], c=labels, s=100)


plt.show()


pca = PCA()
fdata = pca.fit_transform(blobs)

print pca.explained_variance_ratio_

fig = plt.figure()

plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.show()

#

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], c=labels,s=100)

plt.show()


# ISOMAP

from sklearn.manifold import Isomap
iso = Isomap(n_components=3, n_neighbors=30)
fdata = iso.fit_transform(blobs)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], c=labels, s=100)

plt.show()

# LLE

from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_neighbors=15, n_components=3, method='modified')
fig = plt.figure()
fdata = lle.fit_transform(blobs)
ax = fig.add_subplot(111, projection='3d')

plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], c=labels, s=100)

plt.show()

# MDS

from sklearn.manifold import MDS

mds = MDS(n_components=3)
fig = plt.figure()
fdata = mds.fit_transform(blobs)
ax = fig.add_subplot(111, projection='3d')

plt.scatter(fdata[:, 0], fdata[:, 1], zs= fdata[:, 2], c=labels, s=100)

plt.show()
