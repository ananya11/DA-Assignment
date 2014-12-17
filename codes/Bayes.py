# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 22:56:18 2014

@author: vivekbharathakupatni
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:58:34 2014

@author: vivekbharathakupatni
my path = /Users/vivekbharathakupatni/personal/acads/vtech/Fall-2014/Data-Analytics-1/project/DA-Assignment/input


"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.metrics import f1_score



''' Conversion background = 0 
        signal = 1

'''

'''
    IMPORTANT
'''
tmp = pd.read_csv('large.csv', index_col='EventId')

'''
Crosss validation Metric
'''
def getCrossValidationMetric(data, data_labels, metric):
    stFold = cross_validation.KFold(data_labels, n_folds=5)
    gnb = GaussianNB()
    y_metric = cross_validation.cross_val_score(gnb, data, data_labels, 
                                                cv= stFold,
                                                scoring=metric)
    print(y_metric)
    print("length = ",len(y_metric), "Metric =", metric)
    return round(np.mean(y_metric),5)
    
    

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
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=1212)
print("Removing colinearlity on data")

add = pd.Index([])
remove = pd.Index([])
df = pd.DataFrame(data = train_data, columns = tmp.columns - ignore_idx)
corr = df.corr()

for c in corr.columns:
    if c not in remove:
        add = add.union(pd.Index([c]))
    redundant = corr[c][corr[c].abs() > 0.8].index - pd.Index([c]) - add
    #print("adding following indices ", add)
    remove = remove.union(redundant)
    

print(remove)
print(add)

train_data = pd.DataFrame(data=train_data, columns = df.columns)[df.columns- remove].values
test_data = pd.DataFrame(data=test_data, columns = df.columns)[df.columns- remove].values
print("num of featurs = ", train_data.shape[1])


clf = GaussianNB()

# This gets the time in ipython shell.
print("\n\nModelling time:")
%time clf.fit(train_data, train_labels)
print("Modelling time ends\n\n")

print(" Accuracy = " , clf.score(test_data, test_labels))
print(classification_report(test_labels, clf.predict(test_data)))


print("\n\nprediction time starts:")
%time clf.predict(test_data)
print("prediction time ends:\n\n")


'''
Now do the Cross validation score
 
'''

data = tmp[add].values
data_labels = tmp[label_idx].values.reshape(len(tmp))


#print(getCrossValidationMetric(data, data_labels, 'precision'));


''' Verification
'''
'''
from sklearn import cross_validation
X = data
Y = data_labels
kf = cross_validation.KFold(len(X), n_folds=5)

print(kf)  
cross_validation.KFold(n=len(X), n_folds=5, shuffle=False,
                               random_state=None)

for train_index, test_index in kf:
    clf = GaussianNB()
    clf.fit(X[train_index], Y[train_index])
    print("score ", clf.score(data[test_index], labels[test_index]))
    y_pred = clf.predict(data[test_index])
    y_true = labels[test_index] 
    print(f1_score(labels[test_index], y_pred, average='macro'))
    print(f1_score(y_true, y_pred, average='micro'))
    print(f1_score(y_true, y_pred, average='weighted'))
    print(f1_score(y_true, y_pred, average=None))  
    
'''