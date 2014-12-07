# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 22:56:18 2014

@author: vivekbharathakupatni
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:58:34 2014

@author: vivekbharathakupatni
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split


''' Conversion background = 0 
        signal = 1

'''

tmp = pd.read_csv('training.csv', index_col='EventId')


def labelTonumber(l):
    if l == 's':
        return 1
    else:
        return 0


tmp['Label'] = tmp['Label'].map(labelTonumber)

ignore_idx = pd.Index(['Label', 'Weight'])

label_idx = pd.Index(['Label'])

# create a data array
data = tmp[tmp.columns - ignore_idx].values

#create a labels array
labels = tmp[label_idx].values.reshape(len(tmp))




# split into training and test data
# 80 - 20 split.
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

clf = GaussianNB()
clf.fit(train_data, train_labels)

print(" Accuracy = " , clf.score(test_data, test_labels))