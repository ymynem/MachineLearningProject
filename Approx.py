from collections import Counter

def Sorting(n,x,inputfile):
    f=open(inputfile,'r')
    S=f.read()
    mapping = [ ('.', ''), (',', ''), ('?', ''), ('!', ''), ('%', ''),('\n','') ]
    for i,j in mapping:
        S=S.replace(i,j)
    S=S.split(' ')
    subset = [s for s in S if len(s) == n]
    return Counter(subset).most_common(x)