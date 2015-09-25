"""
.. module:: image

image
*************

:Description: image

    

:Authors: bejar
    

:Version: 

:Created on: 23/09/2015 15:19 

"""

__author__ = 'bejar'
from sklearn.datasets import load_sample_image

from sklearn.cluster import DBSCAN, KMeans
from sklearn.cluster import SpectralClustering, AffinityPropagation
from pylab import *
import numpy as np
import matplotlib.cm as cm
from sklearn.mixture import GMM

colors = 'rgbymc'
flower = load_sample_image('flower.jpg')

mdata = np.zeros((flower.shape[0]*flower.shape[1], flower.shape[2]+2))

cc = 0
for i in range(flower.shape[0]):
    for j in range(flower.shape[1]):
        mdata[cc][0] = i
        mdata[cc][1] = j
        for k in range(flower.shape[2]):
            mdata[cc][2+k] = flower[i, j, k]
        cc += 1

plt.figure(figsize=(10,10))
plt.scatter(mdata[:, 0], mdata[:, 1], c=mdata[:, 2:]/255.0, s=1, marker='+')

plt.show()


# K-means
km = KMeans(n_clusters=10, n_jobs=-1)

labels = km.fit_predict(mdata)
plt.figure(figsize=(10,10))
plt.scatter(mdata[:, 0], mdata[:, 1], c=np.array(labels)/len(np.unique(labels)), s=2, marker='+')
plt.show()

# Only Colors
km = KMeans(n_clusters=25, n_jobs=-1)

labels = km.fit_predict(mdata[:,2:])

ccent = km.cluster_centers_/255.0
lcols = [ccent[l,:] for l in labels]
plt.figure(figsize=(10,10))
plt.scatter(mdata[:, 0], mdata[:, 1], c=lcols, s=2, marker='+')
plt.show()


# DBSCAN
dbs = DBSCAN(eps=25, min_samples=50)
print('Done!')
labels = dbs.fit_predict(mdata)
unq = len(np.unique(labels))
ecolors = np.array(labels)
ecolors[ecolors != -1] += 100
ecolors[ecolors == -1] += 0
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121)
plt.scatter(mdata[:, 0], mdata[:, 1], c=mdata[:, 2:]/255.0, s=1, marker='+')

ax = fig.add_subplot(122)
plt.scatter(mdata[:, 0], mdata[:, 1], c=ecolors/unq+100, s=2, marker='+')
plt.show()

