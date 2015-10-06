"""
.. module:: ScalableClustering

ScalableClustering
*************

:Description: ScalableClustering

    

:Authors: bejar
    

:Version: 

:Created on: 06/10/2015 7:59 

"""

__author__ = 'bejar'


from sklearn.cluster import Birch, MiniBatchKMeans, KMeans
import numpy as np
import time
from sklearn.metrics import adjusted_mutual_info_score
from amltlearn.datasets import make_blobs
from pylab import *

ltimes = []
agree = []
step = 10000
limit = 100001
ncenters = 50
for ns in range(step, limit, step):
    blobs, labels = make_blobs(n_samples=ns, n_features=20, centers=ncenters, cluster_std=1, center_box=(-3,3))
    print('NEX= %d'% ns)
    # KMeans
    km = KMeans(n_clusters=ncenters, n_init=1)
    itime = time.perf_counter()
    kmlabels = km.fit_predict(blobs)
    etime = time.perf_counter()
    dtime1 = etime-itime
    agg1 = adjusted_mutual_info_score(labels, kmlabels)

    # Minibatch Kmeans
    itime = time.perf_counter()
    mbkm = MiniBatchKMeans(n_clusters=ncenters, batch_size=int(ns/20), n_init=1)
    mbkmlabels = mbkm.fit_predict(blobs)
    etime = time.perf_counter()
    dtime2 = etime-itime
    agg2 = adjusted_mutual_info_score(labels, mbkmlabels)

    # Birch
    itime = time.perf_counter()
    birch = Birch(threshold=5, n_clusters=ncenters, branching_factor=int(ns/20))
    birchlabels = birch.fit_predict(blobs)
    etime = time.perf_counter()
    dtime3 = etime-itime
    agg3 = adjusted_mutual_info_score(labels, birchlabels)

    ltimes.append((dtime1, dtime2, dtime3))
    print ((dtime1, dtime2, dtime3))
    agree.append((agg1, agg2, agg3))
    print((agg1, agg2, agg3))

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(121)
plt.plot(range(step, limit, step), [x for x,_,_ in ltimes], color='r')
plt.plot(range(step, limit, step), [x for _, x,_ in ltimes], color='g')
plt.plot(range(step, limit, step), [x for _, _, x in ltimes], color='b')



ax = fig.add_subplot(122)
plt.plot(range(step, limit, step), [x for x,_,_ in agree], color='r')
plt.plot(range(step, limit, step), [x for _, x,_ in agree], color='g')
plt.plot(range(step, limit, step), [x for _, _, x in agree], color='b')

plt.show()
