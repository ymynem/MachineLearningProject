import numpy as np
from ssk import ssk
from itertools import product
from collections import defaultdict
from random import shuffle


def enumerate_substring(s, alphabet):
    """
    aaa
    aab
    aac
    ..
    aba
    abb
    ..
    baa
    """
    i = 0
    for j in range(len(s)):
       i += (len(alphabet)**(len(s)-j-1))*alphabet.index(s[j])
    return i


def get_S(n):
    return list("".join(c) for c in product(*(["abcdefghijklmnopqrstuvwxyz "]*n)))


def get_K(S):
    return lambda s, t, n, l: approx_ssk(s, t, n, l, S)


def approx_ssk(s, t, n, l, S):
    #print("!!")
    return sum(ssk(s, si, n, l)*ssk(t, si, n, l) for si in S)


def get_S_from_corpus(corpus, n, alphabet="abcdefghijklmnopqrstuvwxyz "):
    fs = defaultdict(lambda: 0)
    for d in corpus:
        for i in range(len(d)-n):
            #index = enumerate_substring(d[i:i+n], alphabet)
            index = d[i:i+n]
            fs[index] += 1
    
    return [(k, v) for k, v in fs.items()]


def get_top_S(corpus, n, count=200, alphabet="abcdefghijklmnopqrstuvwxyz "):
    fs_list = get_S_from_corpus(corpus, n)
    fs_list.sort(key=lambda p: p[1], reverse=True)  # Sort after count
    subs = [p[0] for p in fs_list[:count]]
    return subs


def get_worst_S(corpus, n, count=200, alphabet="abcdefghijklmnopqrstuvwxyz "):
    fs_list = get_S_from_corpus(corpus, n)
    fs_list.sort(key=lambda p: p[1], reverse=False)  # Sort after count
    subs = [p[0] for p in fs_list[:count]]
    return subs


def get_random_S(corpus, n, count=200, alphabet="abcdefghijklmnopqrstuvwxyz "):
    fs_list = get_S_from_corpus(corpus, n)
    shuffle(fs_list)
    subs = [p[0] for p in fs_list[:count]]
    return subs


if __name__ == "__main__":
    assert enumerate_substring("aaa", "abcdef") == 0
    assert enumerate_substring("aab", "abcdef") == 1
    assert enumerate_substring("aac", "abcdef") == 2
    assert enumerate_substring("aba", "abcdef") == 6
    assert enumerate_substring("abb", "abcdef") == 7
    assert enumerate_substring("aff", "abcdef") == 6**2-1
    assert enumerate_substring("baa", "abcdef") == 6**2
    assert enumerate_substring("fff", "abcdef") == 6**3-1

