"""
.. module:: HierarchicalAuthors

HierarchicalAuthors
*************

:Description: HierarchicalAuthors

    

:Authors: bejar
    

:Version: 

:Created on: 18/09/2015 9:35 

"""

__author__ = 'bejar'

from os import listdir
from os.path import join
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pylab import *
import seaborn as sns
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

docpath = '/home/bejar/Data/authors/Auth1/'


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

authors, alabels = authors_data(method, nfeatures)

clust = linkage(authors, method='ward')
fig = plt.figure(figsize=(8,8))
dendrogram(clust)
plt.show()
labels = fcluster(clust, 2, criterion='maxclust')
print(adjusted_mutual_info_score(alabels, labels))

