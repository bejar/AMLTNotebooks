"""
.. module:: Authors

Authors
*************

:Description: Authors

    

:Authors: bejar
    

:Version: 

:Created on: 14/09/2015 8:03 

"""

__author__ = 'bejar'

from os import listdir
from os.path import join
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pylab import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def show_figure(fdata, labels, ticks):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, c=labels,s=100)
    cbar = plt.colorbar(ticks=range(len(ticks)))
    cbar.ax.set_yticklabels(ticks)

    plt.show()

# Put the Data from authors.zip in any directory and change the docpath variable adequately
# Use the Auth1 or the Auth2 datasets

docpath = '/home/bejar/Data/AMLT/Auth1/'


docs = sorted(listdir(docpath))[1:]


# use v[:2] for dataset labels, use [2:-2] for individual authors
labs = [v[:2] for v in docs]
ulabs = sorted(list(set(labs)))
dlabs = {}
for ul, v  in zip(ulabs, range(len(ulabs))):
    dlabs[ul]=v
labels = [dlabs[v] for v in labs]

pdocs = [join(docpath, f) for f in docs]

nfeatures = 100
method = 1

if method == 1: # Features are word counts
    cvec = CountVectorizer(input='filename', stop_words='english', max_features=nfeatures)
elif method == 2: # Features are TF-IDF
    cvec = TfidfVectorizer(input='filename', stop_words='english', max_features=nfeatures)
elif method == 3: # Features are word occurence
    cvec = TfidfVectorizer(input='filename', stop_words='english', max_features=nfeatures, binary=True, use_idf=False, norm=False)

authors = cvec.fit_transform(pdocs)
authors = authors.toarray()

# print(cvec.vocabulary_)


# PCA
print ('PCA')
pca = PCA()
fdata = pca.fit_transform(authors)

print(pca.explained_variance_ratio_)

fig = plt.figure()

plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.show()

show_figure(fdata, labels, ulabs)


# Sparse PCA
print('Sparse PCA')
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=3)
fdata = spca.fit_transform(authors)
show_figure(fdata, labels, ulabs)

# ISOMAP

print('ISOMAP')
from sklearn.manifold import Isomap
iso = Isomap(n_components=3, n_neighbors=7)
fdata = iso.fit_transform(authors)

show_figure(fdata, labels, ulabs)

# LLE
print('LLE')
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_neighbors=7, n_components=3, method='standard')

fdata = lle.fit_transform(authors)

print(lle.reconstruction_error_)

show_figure(fdata, labels, ulabs)


# MDS
print('MDS')
from sklearn.manifold import MDS
mds = MDS(n_components=3)
fdata = mds.fit_transform(authors)
print (mds.stress_)

show_figure(fdata, labels, ulabs)


# Spectral Embedding
print('Spectral Embedding')
from sklearn.manifold import SpectralEmbedding

spec = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_neighbors=15)
fdata = spec.fit_transform(authors)

show_figure(fdata, labels, ulabs)


# Random Projection
print('Random Projection')
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim
print(johnson_lindenstrauss_min_dim(len(labels), eps=0.9))
grp = GaussianRandomProjection(n_components=3)
fdata = grp.fit_transform(authors)

show_figure(fdata, labels, ulabs)


# NMF

print ('Non Negative Matric Factorization')
from sklearn.decomposition import ProjectedGradientNMF

nmf = ProjectedGradientNMF(n_components=3, sparseness='components' )
fdata = nmf.fit_transform(authors)
print(nmf.reconstruction_err_)

show_figure(fdata, labels, ulabs)

