"""
.. module:: Chair

Chair
*************

:Description: Chair

    

:Authors: bejar
    

:Version: 

:Created on: 14/09/2015 12:27 

"""

__author__ = 'bejar'


from sklearn.decomposition import PCA

from pylab import *
from mpl_toolkits.mplot3d import Axes3D

# Put the Data from  wheel.zip in any directory and change the wheelpath variable adequately

wheelpath = '/home/bejar/Data/Wheel/'

wheeldata1 = np.loadtxt(wheelpath + 'HSGR_Angl.csv', delimiter=',', skiprows=1)
wheeldata2 = np.loadtxt(wheelpath + 'HSGR_Angl-DF.csv', delimiter=',')
wheeldata3 = np.loadtxt(wheelpath + 'HSGR_Angl-AF-DF.csv', delimiter=',')


# PCA

print('PCA')
pca = PCA()
fdata = pca.fit_transform(wheeldata1)
fig = plt.figure(figsize=(30,10))
ax = fig.add_subplot(131, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)

fdata = pca.fit_transform(wheeldata2)
ax = fig.add_subplot(132, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)

fdata = pca.fit_transform(wheeldata3)
ax = fig.add_subplot(133, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)
plt.show()



# ISOMAP
from sklearn.manifold import  Isomap
print('ISOMAP')

iso = Isomap(n_neighbors=20, n_components=3)

fdata = iso.fit_transform(wheeldata1)
fig = plt.figure(figsize=(30,10))
ax = fig.add_subplot(131, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)

fdata = iso.fit_transform(wheeldata2)
ax = fig.add_subplot(132, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)

fdata = iso.fit_transform(wheeldata3)
ax = fig.add_subplot(133, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)
plt.show()


# LLE
print('LLE')
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_neighbors=20, n_components=3, method='standard')

fdata = lle.fit_transform(wheeldata1)
fig = plt.figure(figsize=(30,10))
ax = fig.add_subplot(131, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)

fdata = lle.fit_transform(wheeldata2)
ax = fig.add_subplot(132, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)

fdata = lle.fit_transform(wheeldata3)
ax = fig.add_subplot(133, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)
plt.show()


# Spectral Embedding
print('Spectral Embedding')
from sklearn.manifold import SpectralEmbedding

spec = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_neighbors=30)
fdata = spec.fit_transform(wheeldata1)
fig = plt.figure(figsize=(30,10))
ax = fig.add_subplot(131, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)

fdata = spec.fit_transform(wheeldata2)
ax = fig.add_subplot(132, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)

fdata = spec.fit_transform(wheeldata3)
ax = fig.add_subplot(133, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, s=100)
plt.show()
