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

import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.cluster import DBSCAN, KMeans
from pylab import *
import numpy as np
import matplotlib.cm as cm
from sklearn.mixture import GMM
import time

#
colors = 'rgbymc'

citypath = '/home/bejar/Data/City/'

# Data from the City dataset (BCNpos, PARpos, LONpos)
data = 'BCNpos3.csv'

citypos = loadtxt(citypath+data, delimiter=',')


# What is in the dataset
plt.figure(figsize=(10,10))
plt.scatter(citypos[:, 1], citypos[:, 0],  s=1)
plt.show()

fig = plt.figure(figsize=(10,10))

# Looking for a large number of clusters with K-means
km = KMeans(n_clusters=800, n_init=1)
itime = time.perf_counter()
labels = km.fit_predict(citypos)
etime = time.perf_counter()
print (etime-itime)
print(len(np.unique(labels)))


ax = fig.add_subplot(111)
ax.set_title('BCN K-Means')
ax.scatter(citypos[:, 1], citypos[:, 0], c=labels/len(np.unique(labels))*1.0, s=2, marker='+')
plt.show()

# GMM
# gmm = GMM(n_components=30, covariance_type='diag')
# gmm.fit(citypos)
#
# labels = gmm.predict(citypos)
# print(len(np.unique(labels)))
#
# ax = fig.add_subplot(222)
# ax.set_title('GMM')
#
# ax.scatter(citypos[:, 1], citypos[:, 0], c=labels/len(np.unique(labels))*1.0, s=2, marker='+')
#plt.show()

#Leader
from kemlglearn.cluster import Leader

lead = Leader(radius=0.004)
itime = time.perf_counter()
lead.fit(citypos)
labels = lead.predict(citypos)
etime = time.perf_counter()
print (etime-itime)
print(len(np.unique(labels)))
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
ax.set_title('BCN Leader')
ax.scatter(citypos[:, 1], citypos[:, 0], c=np.array(labels)/len(np.unique(labels))*1.0, s=2, marker='+')
plt.show()

#
# Adjusting DBSCAN parameters is tricky
dbs = DBSCAN(eps=0.0005, min_samples=5)
itime = time.perf_counter()
labels = dbs.fit_predict(citypos)
etime = time.perf_counter()
print (etime-itime)
print(len(np.unique(labels)))
labels[labels == -1] += len(np.unique(labels))+10
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
ax.set_title('BCN DBSCAN')
ax.scatter(citypos[:, 1], citypos[:, 0], c=(labels+1)/len(np.unique(labels))*1.0, s=2, marker='+')

plt.show()

from sklearn.cluster import Birch

birch = Birch(threshold=0.002, n_clusters=800, branching_factor=50)

itime = time.perf_counter()
labels = birch.fit_predict(citypos)
etime = time.perf_counter()

print (etime-itime)
print(len(np.unique(labels)))
labels[labels == -1] += len(np.unique(labels))+10
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
ax.set_title('BCN BIRCH')
ax.scatter(citypos[:, 1], citypos[:, 0], c=(labels+1)/len(np.unique(labels))*1.0, s=2, marker='+')

plt.show()
