"""
.. module:: test

test
*************

:Description: test

    

:Authors: bejar
    

:Version: 

:Created on: 18/09/2015 14:55 

"""

__author__ = 'bejar'

from pyclustering.nnet.som import som
from sklearn import datasets


iris = datasets.load_iris()

nsom = som(5,5)

nsom.train(iris['data'][:,:3], epochs=200)

nsom.show_density_matrix()
nsom.show_winner_matrix()
nsom.show_distance_matrix()
nsom.show_network(belongs=True)