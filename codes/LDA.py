# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 20:27:35 2014

@author: vivekbharathakupatni
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 22:56:18 2014

@author: vivekbharathakupatni
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:58:34 2014

@author: vivekbharathakupatni
my path = /Users/vivekbharathakupatni/personal/acads/vtech/Fall-2014/Data-Analytics-1/project/kaggle


for LDA

"""

import pandas as pd
import numpy as np
from sklearn.lda import LDA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


''' Conversion background = 0 
        signal = 1

'''

tmp = pd.read_csv('large.csv', index_col='EventId')


def labelTonumber(l):
    if l == 's':
        return 1
    else:
        return 0


tmp['Label'] = tmp['Label'].map(labelTonumber)

ignore_idx = pd.Index(['Label', 'Weight'])

label_idx = pd.Index(['Label'])


#create a labels array
labels = tmp[label_idx].values.reshape(len(tmp))

# create a data array
data = tmp[tmp.columns - ignore_idx].values





# split into training and test data
# 80 - 20 split.
train_data_g, test_data_g, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

print("Removing colinearlity on data")

df = pd.DataFrame(data = train_data_g, columns = tmp.columns - ignore_idx)
corr = df.corr()

cor_coefficients = [0.8, 0.85, 0.9, 0.95]
for coefficient in cor_coefficients:
    add = pd.Index([])
    remove = pd.Index([])    
    for c in corr.columns:
    #print("column = ", c, corr[c][corr[c] > 0.98].index - pd.Index([c]))
        if c not in remove:
            add = add.union(pd.Index([c]))
        redundant = corr[c][corr[c].abs() > coefficient].index - pd.Index([c]) - add
        remove = remove.union(redundant)
    
    print("For correlation coefficient = ", coefficient)
    #print(remove)
    #print(add)

    train_data = pd.DataFrame(data=train_data_g, columns = df.columns)[df.columns- remove].values
    test_data = pd.DataFrame(data=test_data_g, columns = df.columns)[df.columns- remove].values
    print("num of featurs = ", train_data.shape[1])

    clf = LDA();

    # This gets the time in ipython shell.
    print("\n\nModelling time:")
    %time clf.fit(train_data, train_labels)
    print("Modelling time ends\n\n")

    print("\n\nprediction time starts:")
    %time predicted_labels = clf.predict(test_data)
    print("prediction time ends:\n\n")
    #print(classification_report(test_labels, clf.predict(test_data)))
    print(classification_report(test_labels, predicted_labels))

    print("num of featurs = ", train_data.shape[1])
    y_true = test_labels;
    y_pred_proba = clf.predict_proba(test_data);
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    print("ROC AUC =", roc_auc)
