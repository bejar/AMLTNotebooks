"""
.. module:: text

text
*************

:Description: text

    

:Authors: bejar
    

:Version: 

:Created on: 08/10/2015 10:06 

"""

__author__ = 'bejar'


from sklearn.feature_extraction.text import CountVectorizer
from os import listdir
from os.path import join
from fim import fpgrowth, apriori

docpath = '/home/bejar/Data/authors/Auth2/'
docs = sorted(listdir(docpath))[1:]
pdocs = [join(docpath, f) for f in docs]
nfeatures = 200
cvec = CountVectorizer(input='filename', stop_words='english', max_features=nfeatures)
authors = cvec.fit_transform(pdocs)

lwords = [""] * len(cvec.vocabulary_)
for w in cvec.vocabulary_:
    lwords[cvec.vocabulary_[w]] = w

ldocs = []
for j in range(authors.shape[0]):
    doc = []
    for i in range(authors.shape[1]):
        if authors[j,i] != 0:
            doc.append(lwords[i])
    ldocs.append(doc)

res = fpgrowth(ldocs, target='s', supp=10, zmin=1, report='ae', eval='c', algo='t')

for i in range(10):
    print(res[i])

print(len(res))


res = apriori(ldocs, target='s', supp=10, zmin=1, report='ae', eval='c')

for i in range(10):
    print(res[i])

print(len(res))


