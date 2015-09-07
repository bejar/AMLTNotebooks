"""
.. module:: DenseSubgraphs

DenseSubgraphs
*************

:Description: DenseSubgraphs

    

:Authors: bejar
    

:Version: 

:Created on: 03/09/2015 11:16 

"""

__author__ = 'bejar'


import networkx as nx
import matplotlib.pyplot as plt
import folium
from geojson import LineString, FeatureCollection, Feature
import geojson


fname = './Data/paristwitter.txt'


rfile = open(fname, 'r')

gr = nx.Graph()

for lines in rfile:
    vals = lines.replace('[', '').replace(']','').replace('\n','').replace('\'','').replace(' ','').split(',')
    #print vals
    for v1 in vals:
        for v2 in vals:
            if v1 != v2:
                gr.add_edge(v1,v2)

coord = (48.52, 49.05, 1.97, 2.68) #(51.23, 51.8, -0.5, 0.37) #(41.20, 41.65, 1.90, 2.40)


# nx.draw_spring(gr)
# plt.show()

a = nx.find_cliques(gr)

#print len(list(a))

m=0
for c in a:
     if len(c)>10:
        print len(c)
        m += 1
        seq = []
        for i in c:
            x, y, h = i.split('#')
            seq.append((x, y, h))
        lgeo = []
        for s1 in seq:
            for s2 in seq:
                if s1 != s2:
                    x1, y1, _ = s1
                    x2, y2, _ = s2
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    lgeo.append(Feature(geometry=LineString([(y1, x1), (y2, x2)])))
        geoc = FeatureCollection(lgeo)
        dump = geojson.dumps(geoc)
        jsfile = open('/home/bejar/tmp/Map' + str(m) + '-' + str(len(c)) + '.json', 'w')
        jsfile.write(dump)
        jsfile.close()
        mymap = folium.Map(location=[(coord[0] + coord[1]) / 2.0, (coord[2] + coord[3]) / 2.0], zoom_start=12, width=1200,
                           height=1000)
        mymap.geo_json(geo_path='/home/bejar/tmp/Map' + str(m) + '-' + str(len(c)) + '.json', fill_color='Black', line_color='Black',
                       line_weight=2)
        mymap.create_map(path='/home/bejar/tmp/Map' + str(m) + '-' + str(len(c)) + '.html')

# nx.draw_circular(a)
# plt.show()
