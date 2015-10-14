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

#/*--------------------------------------------------------------------*/
#/* apriori (tracts, target='s', supp=10, conf=80, zmin=1, zmax=None,  */
#/*          report='a', eval='x', agg='x', thresh=10, prune=None,     */
#/*          algo='', mode='', border=None, appear=None)               */
#/*--------------------------------------------------------------------*/

print  ('apriori(tracts, supp=-3, zmin=2):')
for r in apriori(tracts, target='r', supp=-3, zmin=2, report='asicC'): print(r)


#/*--------------------------------------------------------------------*/
#/* fpgrowth (tracts, target='s', supp=10, conf=80, zmin=1, zmax=None, */
#/*           report='a', eval='x', agg='x', thresh=10, prune=None,    */
#/*           algo='s', mode='', border=None, appear=None)             */
#/*--------------------------------------------------------------------*/

print  ('fpgrowth(tracts, supp=-3, zmin=2):')
for r in fpgrowth(tracts, target='r', supp=-3, zmin=2, report='asicC'): print(r)
