import numpy as np

"""
Call this file using calcmean with the name of the file with the data
"""

def mean_SD_calc(filename):
    """
    Extracts every even numbered row in a data file starting with row number 2
    """
    raw_data=[]
    i=1
    f=open(filename)
    for line in f.readlines():
        if i%2==0:
            raw_data.append(line)
        i+=1
        
    return raw_data


def calcmean(filename):
    """
    Returns a vector of mean and a vector of standard deviation for F1, Precision and recall
    
    """
    data=mean_SD_calc(filename)
    raw_data=[]
    
    for i in range(len(data)):
        raw_data.append(data[i].split(" "))
    
    raw_data=np.array(raw_data)
    
    for row in range(len(raw_data)):
        raw_data[row][-1]=raw_data[row][-1].rstrip('\n')
        
    raw_data=raw_data.astype(np.float)
    
    return np.mean(raw_data,axis=0), np.std(raw_data,axis=0)