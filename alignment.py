import numpy as np
from utils import write_data
from approx import approx_ssk, get_top_S, get_worst_S, get_random_S, get_K
from bgm import build_gram_matrix as bgm
from simple_data import DATA


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


def get_alignment(K, corpus, n, l, S):
    k = get_K(S)
    return alignment(K, bgm(corpus, corpus, l, n, K=k))


def compare_approx():
    corpus = DATA[0]["train"]["x"]
    n = 3
    l = 0.5
    print("computing real gram, this might take awile...")
    real_gram = bgm(corpus, corpus, l, n)
    print("rest should be done in a moment")
    aligns = []
    M = 1000
    m = 10
    S = get_top_S(corpus, n, count=M)
    S_ = get_worst_S(corpus, n, count=M)
    S__ = get_random_S(corpus, n, count=M)
    for i in range(1, M+1, 10):
        aligns.append((
            get_alignment(real_gram, corpus, n, l, S[:i]),
            get_alignment(real_gram, corpus, n, l, S_[:i]),
            get_alignment(real_gram, corpus, n, l, S__[:i]),
        ))
    
    print(aligns[:100])
    write_data("aligns-3.json", aligns)


if __name__ == "__main__":
    compare_approx()

