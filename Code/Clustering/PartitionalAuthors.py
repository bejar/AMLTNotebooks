"""
.. module:: PartitionalAuthors

PartitionalAuthors
*************

:Description: PartitionalAuthors

    

:Authors: bejar
    

:Version: 

:Created on: 18/09/2015 9:58 

"""

__author__ = 'bejar'

from os import listdir
from os.path import join
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pylab import *
import seaborn as sns
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

docpath = '/home/bejar/Data/authors/Auth1/'


def show_figure(fdata, labels1, labels2, title='', subtitle=['','']):
    fig = plt.figure(figsize=(12,10))
    fig.suptitle(title, fontsize=32)
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(subtitle[0])
    plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, c=labels1,s=100)
    # cbar = plt.colorbar(ticks=range(len(ticks)))
    # cbar.ax.set_yticklabels(ticks)
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(subtitle[1])
    plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], depthshade=False, c=labels2,s=100)

    plt.show()

def authors_data(method=1, nfeatures=100):
    """
    Returns a data matrix representing the documents using the indicated method for generating the features and the
    specified number of features and the labels for the documents

    :param method:
    :param nfeatures:
    :return:
    """

    docs = sorted(listdir(docpath))[1:]

    # use v[:2] for dataset labels, use [2:-2] for individual authors
    labs = [v[:2] for v in docs]
    ulabs = sorted(list(set(labs)))
    dlabs = {}
    for ul, v  in zip(ulabs, range(len(ulabs))):
        dlabs[ul]=v
    labels = [dlabs[v] for v in labs]

    pdocs = [join(docpath, f) for f in docs]



    if method == 1: # Features are word counts
        cvec = CountVectorizer(input='filename', stop_words='english', max_features=nfeatures)
    elif method == 2: # Features are TF-IDF
        cvec = TfidfVectorizer(input='filename', stop_words='english', max_features=nfeatures)
    elif method == 3: # Features are word occurence
        cvec = TfidfVectorizer(input='filename', stop_words='english', max_features=nfeatures, binary=True, use_idf=False, norm=False)

    return cvec.fit_transform(pdocs).toarray(), labels

nfeatures = 1500
method = 2
nclusters = 2

authors, alabels = authors_data(method, nfeatures)

pca = PCA()
fdata = pca.fit_transform(authors)


# KMeans
km = KMeans(n_clusters=nclusters, n_init=10, random_state=0)
labels = km.fit_predict(authors)

print(adjusted_mutual_info_score(alabels, labels))

show_figure(fdata, alabels, labels, subtitle=['Orig','K-Means'])

# GMM
# covariance_type = spherical, tied, diag, full
gmm = GMM(n_components=nclusters, covariance_type='diag', random_state=0)
gmm.fit(authors)

labels = gmm.predict(authors)

print(adjusted_mutual_info_score(alabels, labels))

show_figure(fdata, alabels, labels, subtitle=['Orig','GMM'])

# Spectral Clustering
spec = SpectralClustering(n_clusters=nclusters, affinity='nearest_neighbors', n_neighbors=15, random_state=0)
labels = spec.fit_predict(authors)

print(adjusted_mutual_info_score(alabels, labels))

show_figure(fdata, alabels, labels, subtitle=['Orig','Spectral'])

