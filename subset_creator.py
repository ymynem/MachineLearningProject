from collections import Counter
from reuters import *
import re
"""
Approximating string kernel using the most common occurring
strings in documents. This function creates a subset of the full
set of words in each document.

Example:
    most_common(4,5,'corn')
    returns the five most common 4 length words occurring in dataset 'corn'
    
Author: Fabian Huss
"""

def sorting(n,training, testing):
    """
    n: length of string/word for subset
    category: which category in reuters to be used
    
    Returns a list of x most frequently occuring n length characters in a file
    """
    
    ltrain=len(training)
    ltest=len(testing)
    train_sub=[]
    test_sub=[]
    test=[]
    train=[]
    for i in range(ltrain):
        train.append(training[i].split())
    for j in range(ltest):
        test.append(testing[j].split())
    
    for i in range(len(train)):
        train_sub.append([s for s in train[i] if len(s)==n])
    for j in range(len(test)):
        test_sub.append([s for s in test[j] if len(s)==n])
    
    return train_sub, test_sub

def list_to_string(train,test):
    
    training=[]
    testing=[]
    for i in range(len(train)):
        training.append(' '.join(train[i]))
    for j in range(len(test)):
        testing.append(' '.join(test[j]))
    return training, testing

def most_common(n,x,train,test):
    """
    x: how many most frequently occurring n length words
    
    """
    train, test = sorting(n,train, test)
    
    train_cmn=[]
    test_cmn=[]
    ltrain=len(train)
    ltest=len(test)
    
    """
    appends the x most frequently used n length words to train_cmn/test_cmn
    for each document
    """
    for i in range(ltrain):
        train_cmn.append(Counter(train[i]).most_common(x))
    for j in range(ltest):
        test_cmn.append(Counter(test[j]).most_common(x))
    for i in range(ltrain):
        train_cmn[i]=[x[0] for x in train_cmn[i]]
    for j in range(ltest):
        test_cmn[j]=[x[0] for x in test_cmn[j]]
    
    train_cmn,test_cmn=list_to_string(train_cmn,test_cmn)
    
        
    return train_cmn,test_cmn
    

    
    
