# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 01:14:30 2014

@author: vivekbharathakupatni
"""


from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


def augmented_dendrogram(*args, **kwargs):

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')

    return ddata
    


import numpy as np
from scipy.cluster.hierarchy import linkage



x = np.array([ [1, 0, 1, 1, 0],
     [1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [0,1,0,1,0],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0]])
    
plt.figure(1, figsize=(6, 5))
plt.clf()
plt.scatter(x[:, 0], x[:, 1])
plt.axis('equal')
plt.grid(True)


def funRC(a, b):
    n_00 = 0
    n_01 = 0
    n_10 = 0
    n_11 = 0
    for i,j in zip(a,b):
        if i==0 and j ==0:
            n_00 += 1
        elif i==0 and j==1:
            n_01 += 1
        elif i==1 and j == 0:
            n_10 += 1
        elif i==1 and j == 1:
            n_11 += 1
    
    '''print("a" , a)
    print("b", b)
    print("n11", n_11)
    print("n00", n_00)
    print("sum = ",n_11+n_10 + n_01 + n_00)'''
    res = ((n_11*1.0)/(n_11+n_10 + n_01 + n_00))
    #print("res", res)
    return res

    
linkage_matrix = linkage(x, "single", funRC)

plt.figure(2, figsize=(10, 4))
plt.clf()

plt.subplot(1, 2, 1)
show_leaf_counts = False
ddata = augmented_dendrogram(linkage_matrix,
               color_threshold=1,
               p=6,
               truncate_mode='lastp',
               show_leaf_counts=show_leaf_counts,
               labels=np.array(['x1','x2','x3', 'x4', 'x5', 'x6'])
               )
plt.title("show_leaf_counts = %s" % show_leaf_counts)

plt.subplot(1, 2, 2)
show_leaf_counts = True
ddata = augmented_dendrogram(linkage_matrix,
               color_threshold=1,
               p=6,
               truncate_mode='lastp',
               show_leaf_counts=show_leaf_counts,
               labels=np.array(['x1','x2','x3', 'x4', 'x5', 'x6']),
               )
plt.title("show_leaf_counts = %s" % show_leaf_counts)

plt.show()


# Get the RC
res = np.zeros((6,6))
for i in range(len(x)):
    for j in range(len(x)):
        res[i][j] = funRC(x[i], x[j])
'''
'''
def funSMC(a, b):
    n_00 = 0
    n_01 = 0
    n_10 = 0
    n_11 = 0   
    for i,j in zip(a,b):
        if i==0 and j ==0:
            n_00 += 1
        elif i==0 and j==1:
            n_01 += 1
        elif i==1 and j == 0:
            n_10 += 1
        elif i==1 and j == 1:
            n_11 += 1
    num = (n_11 + n_00) * 1.0
    den = (n_11+n_10 + n_01 + n_00) * 1.0
    
    return num/den
   
    
# Gte the smc
smc = np.zeros((6,6))
for i in range(len(x)):
    for j in range(len(x)):
        smc[i][j] = funSMC(x[i], x[j])

'''    
        
        
# part b problem

'''
linkage_matrix = linkage(x, "complete", funSMC)

plt.figure(2, figsize=(10, 4))
plt.clf()

plt.subplot(1, 2, 1)
show_leaf_counts = False
ddata = augmented_dendrogram(linkage_matrix,
               color_threshold=1,
               p=6,
               truncate_mode='lastp',
               show_leaf_counts=show_leaf_counts,
               )
plt.title("show_leaf_counts = %s" % show_leaf_counts)

plt.subplot(1, 2, 2)
show_leaf_counts = True
ddata = augmented_dendrogram(linkage_matrix,
               color_threshold=1,
               p=6,
               truncate_mode='lastp',
               show_leaf_counts=show_leaf_counts,
               )
plt.title("show_leaf_counts = %s" % show_leaf_counts)

plt.show()
'''




# part c avg with JC

def funJC(a, b):
    n_00 = 0
    n_01 = 0
    n_10 = 0
    n_11 = 0   
    for i,j in zip(a,b):
        if i==0 and j ==0:
            n_00 += 1
        elif i==0 and j==1:
            n_01 += 1
        elif i==1 and j == 0:
            n_10 += 1
        elif i==1 and j == 1:
            n_11 += 1
    num = (n_11) * 1.0
    den = (n_11+n_10 + n_01) * 1.0
    
    return num/den

linkage_matrix = linkage(x, "average", funJC)

plt.figure(2, figsize=(10, 4))
plt.clf()

plt.subplot(1, 2, 1)
show_leaf_counts = False
ddata = augmented_dendrogram(linkage_matrix,
               color_threshold=1,
               p=6,
               truncate_mode='lastp',
               show_leaf_counts=show_leaf_counts,
               )
plt.title("show_leaf_counts = %s" % show_leaf_counts)

plt.subplot(1, 2, 2)
show_leaf_counts = True
ddata = augmented_dendrogram(linkage_matrix,
               color_threshold=1,
               p=6,
               truncate_mode='lastp',
               show_leaf_counts=show_leaf_counts,
               labels=np.array(['x1','x2','x3', 'x4', 'x5', 'x6']))
plt.title("show_leaf_counts = %s" % show_leaf_counts)

plt.show()


# distance function for 3 part
jc = np.zeros((6,6))
for i in range(len(x)):
    for j in range(len(x)):
        jc[i][j] = funJC(x[i], x[j])
        
        

def findAvg(*args):
    avg = 0   
    for x in args:
        avg += x
    
    return (avg * 1.0)/ len(args)