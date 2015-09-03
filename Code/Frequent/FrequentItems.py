"""
.. module:: FrequentItems

FrequentItems
*************

:Description: FrequentItems

    

:Authors: bejar
    

:Version: 

:Created on: 03/09/2015 10:28 

"""

__author__ = 'bejar'

from fim import apriori, eclat, fpgrowth, fim


tracts = [ [ 1, 2, 3 ],
           [ 1, 4, 5 ],
           [ 2, 3, 4 ],
           [ 1, 2, 3, 4 ],
           [ 2, 3 ],
           [ 1, 2, 4 ],
           [ 4, 5 ],
           [ 1, 2, 3, 4 ],
           [ 3, 4, 5 ],
           [ 1, 2, 3 ] ]

print('transactions:')
for t in tracts: print(t)

print  ('apriori(tracts, supp=-3, zmin=2):')
for r in apriori(tracts, supp=-3, zmin=2): print r

print  ('eclat(tracts, supp=-3, zmin=2):')
for r in eclat(tracts, supp=-3, zmin=2): print r

print  ('fpgrowth(tracts, supp=-3, zmin=2):')
for r in fpgrowth(tracts, supp=-3, zmin=2,): print r

print  ('fim(tracts, supp=-3, zmin=2, report=\'#\'):')
for r in fim(tracts, supp=-3, zmin=2, report='#'): print r
