from collections import Counter

def Sorting(n,x,inputfile):
    """
    n = length of string/word needed for subset
    x = how many most frequently occuring n length words
    inputfile = dataset used for algorithm
    
    Returns a list of x most frequently occuring n length caracters in a file
    """
    f=open(inputfile,'r')
    S=f.read().lower()
    mapping = [ ('.', ''), (',', ''), ('?', ''), ('!', ''), ('%', ''),('\n',''),('\t','')]
    for i,j in mapping:
        S=S.replace(i,j)
    S=S.split(' ')
    subset = [s for s in S if len(s) == n]
    mst_cmn=Counter(subset).most_common(x) 
    mst_cmn=[x[0] for x in mst_cmn]
    
    return mst_cmn
    