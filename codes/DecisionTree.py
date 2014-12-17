# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 02:24:23 2014

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

"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import re
from sklearn.preprocessing import Imputer


def getMEanMissingData(X):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X.shape)
    df = pd.DataFrame(X)
    df[df == -999] = np.nan
    print(df)
    return imp.transform(df.values)
    

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

'''
    Handle missing data
'''

ignore_idx = pd.Index(['Label', 'Weight'])

label_idx = pd.Index(['Label'])

# create a data array
data = tmp[tmp.columns - ignore_idx].values

#create a labels array
labels = tmp[label_idx].values.reshape(len(tmp))


opt_criterion = None
opt_min_split = None
opt_f1_score = None

# Original value used before finding correlation
#min_splits = range(1, 10, 1) + range(10, 100, 10) + range(100, 1000, 100) + range(1000, 10000, 1000)

#new value 
min_splits = range(100, 600, 100)

def extract_f1_score(text):
    index = text.find('avg')
    avg_text = text[index:].strip()
    return float(re.split(r'\s+', avg_text)[5])
    
    
# split into training and test data
# 80 - 20 split.

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
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

'''
for crit in ['gini', 'entropy']:
    
    clf = tree.DecisionTreeClassifier(criterion=crit, min_samples_split=1)

    print("Results for criterian = ", crit)    
    # This gets the time in ipython shell.
    print("\n\nModelling time:")
    %time clf.fit(train_data, train_labels)
    print("Modelling time ends\n\n")

    #print(" Accuracy = " , clf.score(test_data, test_labels))
    prediction =  clf.predict(test_data)   
    print(classification_report(test_labels, prediction))
    print("extracting f1 score = ", extract_f1_score(classification_report(test_labels, prediction)))

    print("\n\nprediction time starts:")
    %time clf.predict(test_data)
    print("prediction time ends:\n\n")
    
    print("Results for criterian = ", crit, "ENDS****") 
    
'''

for crit in ['gini']:
    
    for min_samples in  min_splits:
        print("\n\n\nRunning code for criterian = ", crit, " min samples ", min_samples)           
        clf = tree.DecisionTreeClassifier(criterion=crit, min_samples_split=min_samples)

         
        # This gets the time in ipython shell.
        print("Modelling started:")
        clf.fit(train_data, train_labels)
        print("Modelling ended:")
        
        prediction =  clf.predict(test_data)   
        print(classification_report(test_labels, prediction))
        f1 = extract_f1_score(classification_report(test_labels, prediction))        
        print("extracting f1 score = ", f1)
    
        print("code for criterian = ", crit, " min samples ", min_samples, "ends") 
        if opt_min_split == None:
            opt_min_split = min_samples
            opt_criterion = crit
            opt_f1_score = f1
        else:
            if opt_f1_score < f1:
                opt_f1_score = f1
                opt_min_split = min_samples
                opt_criterion = crit
