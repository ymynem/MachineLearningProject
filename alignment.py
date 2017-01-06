import numpy as np

def alignment(K1,K2):
    """
    Measures the alignment according
    to the frobenius inner product
    of two matrices
    """
    assert np.shape(K1) == np.shape(K2)
    K1 = np.array(K1)
    K2 = np.array(K2)
    algn = np.sum(K1*K2)/np.sqrt(np.sum(K1*K1)*np.sum(K2*K2))
    
    return algn