from math import sqrt
from functools import lru_cache

import sys
sys.setrecursionlimit(10000)  # This is bad practice


@lru_cache(maxsize=None)
def kh(s, t, n, l):
    @lru_cache(maxsize=None)
    def kmm(n, si, ti):
         if n == 0:
             return 1
         if min(si, ti) < n:
             return 0
         if s[si-1] == t[ti-1]:
             return l*(kmm(n, si, ti-1) + l*km(n-1, si-1, ti-1))
         else:
             return l * kmm(n, si, ti-1)
#         return sum(km(n-1, si-1, j) * l**(ti-(j+1)+2) for j in range(ti) if t[j] == s[si-1])

    @lru_cache(maxsize=None)
    def km(n, si, ti):
        if n == 0:
            return 1
        if min(si, ti) < n:
            return 0
        return l*km(n, si-1, ti) + kmm(n, si, ti)
#        return l*km(n, si-1, ti) + sum(km(n-1, si-1, j) * l**(ti-(j+1)+2) for j in range(ti) if t[j] == s[si-1])

    @lru_cache(maxsize=None)
    def k(n, si, ti):
        if min(si, ti) < n:
            return 0
        return k(n, si-1, ti) + sum(km(n-1, si-1, j) for j in range(ti) if t[j] == s[si-1]) * l**2

    return k(n, len(s), len(t))

def ssk(s, t, n, l):
    return kh(s, t, n, l) / sqrt(kh(s, s, n, l) * kh(t, t, n, l))



