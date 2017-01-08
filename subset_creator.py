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

def acquire_list(category):
    """
    Acquires corpus from reuters dataset and returns as list of strings
    """
    train_data=create_corpus(get_documents(category)[0])
    test_data=create_corpus(get_documents(category)[1])
    
    test=[]
    train=[]
    
    ltrain=len(train_data)
    ltest=len(test_data)
    
    for i in range(ltrain):
        train.append(re.sub('[^a-z ]+', '', train_data[i]))
    for j in range(ltrain):
        train[j]=train[j].split()
    for i in range(ltest):
        test.append(re.sub('[^a-z ]+', '', test_data[i]))
    for j in range(ltest):
        test[j]=test[j].split()
    return train,test

def sorting(n,category):
    """
    n: length of string/word for subset
    category: which category in reuters to be used
    
    Returns a list of x most frequently occuring n length caracters in a file
    """
    train,test = acquire_list(category)
    
    train_sub=[]
    test_sub=[]
    ltrain=len(train)
    ltest=len(test)
    for i in range(ltrain):
        train_sub.append([s for s in train[i] if len(s)==n])
    for j in range(ltest):
        test_sub.append([s for s in test[j] if len(s)==n])
    
    return train_sub, test_sub

def most_common(n,x,category):
    """
    x: how many most frequently occurring n length words
    
    """
    train, test =sorting(n,category)
    
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
        
        
    return train_cmn,test_cmn
    
    
