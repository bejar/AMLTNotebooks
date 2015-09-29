"""
.. module:: DensityBased

DensityBased
*************

:Description: DensityBased

    

:Authors: bejar
    

:Version: 

:Created on: 23/09/2015 10:03 

"""

__author__ = 'bejar'

from numpy import loadtxt
from sklearn.cluster import DBSCAN, KMeans
from pylab import *
import numpy as np
import matplotlib.cm as cm
from sklearn.mixture import GMM

#
colors = 'rgbymc'

citypath = '/home/bejar/Data/City/'

# Data from the City dataset (BCNpos, PARpos, LONpos)
data = 'PARpos.csv'

citypos = loadtxt(citypath+data, delimiter=',')


# What is in the dataset
plt.figure(figsize=(10,10))
plt.scatter(citypos[:, 1], citypos[:, 0],  s=1)
plt.show()

# Looking for a large number of clusters with K-means
km = KMeans(n_clusters=30)

labels = km.fit_predict(citypos)
print(len(np.unique(labels)))


fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(221)
ax.scatter(citypos[:, 1], citypos[:, 0], c=labels/len(np.unique(labels))*1.0, s=2, marker='+')


# GMM
gmm = GMM(n_components=30, covariance_type='diag')
gmm.fit(citypos)

labels = gmm.predict(citypos)
print(len(np.unique(labels)))

ax = fig.add_subplot(222)
ax.scatter(citypos[:, 1], citypos[:, 0], c=labels/len(np.unique(labels))*1.0, s=2, marker='+')

# Leader
from amltlearn.cluster import Leader

lead = Leader(radius=0.04)

lead.fit(citypos)
labels = lead.predict(citypos)
print(len(np.unique(labels)))

ax = fig.add_subplot(223)
ax.scatter(citypos[:, 1], citypos[:, 0], c=np.array(labels)/len(np.unique(labels))*1.0, s=2, marker='+')


# Adjusting DBSCAN parameters is tricky
dbs = DBSCAN(eps=0.005, min_samples=75)
labels = dbs.fit_predict(citypos)
print(len(np.unique(labels)))
labels[labels == -1] += len(np.unique(labels))+10

ax = fig.add_subplot(224)
ax.scatter(citypos[:, 1], citypos[:, 0], c=(labels+1)/len(np.unique(labels))*1.0, s=2, marker='+')


plt.show()

