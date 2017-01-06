# -*- coding: utf-8 -*-
""" page 424
"""
import math
import cProfile
from functools import lru_cache, partial


def subsequence_kernel_double_primed(s, t, l, i):
    if i == 0:
        return 1
    elif min(len(s), len(t)) < i:
        return 0
    else:
        x = s[-1]  # last character. sx means the hole string, when they write s they mean exclude last char
        the_sum = 0
        for j in range(len(t)):
            if t[-1] == s[-1]:
              #  res = l * subsequence_kernel_double_primed(s[:-1], t, l, i) + l * subsequence_kernel_primed_lru_wrapped(s[:-1], t,
              #                                                                                              l, i)
                return res


def subsequence_kernel_primed_lru_wrapped(s, t):
    @lru_cache(maxsize=None)
    def subsequence_kernel_primed(s_counter, jtot, l, i):  # where (i = 1, … , n-1)
        """
        In order to deal with non-contiguous substrings, it is necessary to
        introduce a decay factor λ ∈ (0, 1) that can be used to weight the presence of a certain feature in a text
        :param s: string 1
        :param t: string 2
        :param l: lambda represents the weight?
        :param i: length of subsequence
        :return:
        """
        if i == 0:
            return 1
        elif min(s_counter, jtot) < i:  #
            return 0
        else:
            x = s[s_counter - 1]  # last character. sx means the hole string, when they write only s they mean exclude last char
            the_sum = 0

            for j in range(jtot):
                if x == t[j]:
                    the_sum += subsequence_kernel_primed_lru_wrapped(s, t)(s_counter - 1, j, l, i - 1) * l ** (jtot - j + 2)
        res = l * subsequence_kernel_primed_lru_wrapped(s, t)(s_counter - 1, jtot, l, i) + the_sum
        return res

    return subsequence_kernel_primed


def subsequence_kernel(s, t, l, n):  # where (i = 1, … , n-1)
    """
    In order to deal with non-contiguous substrings, it is necessary to
    introduce a decay factor λ ∈ (0, 1) that can be used to weight the presence of a certain feature in a text
    :param s: string 1
    :param t: string 2
    :param l: lambda represents the weight?
    :param n: length of string
    :return:
    """
    s_len = len(s)
    if min(s_len, len(t)) < n:
        return 0
    else:
        the_sum = 0
        if n > 0:
            x = s[-1]
            for j in range(len(t)):
                if t[j] == x:
                    the_sum += subsequence_kernel_primed_lru_wrapped(s, t)(s_len - 1, j, l, n - 1) * l ** 2  # [:-1]
    res = subsequence_kernel(s[:-1], t, l, n) + the_sum
    return res


def normalize(s, t, l, n):
    norm = subsequence_kernel(s, t, l, n) / math.sqrt(subsequence_kernel(s, s, l, n) * subsequence_kernel(t, t, l, n))
    return norm



def main():
    """
  s = string
  u = sing
  I = [0,3,4,5]
  u = s[I]
  u = s[0]s[3]s[4]s[5]
  |u| = 4
  L(I) = i_|u| - i_1 + 1
  L(I) = 5 - 0 + 1 = 6
    """

    #   print("1: ", subsequence_kernel_primed("car", "cat", 0.5, 2))
    # s = "Anyone who reads Old and Middle English"
    # t = "Πόθεν, ὦ Σώκρατες, φαίνῃ; Ἤ δῆλα δὴ ὅτι"
    l = 0.5
    n = 2
    s_glob = "and normal sized documents, direct "
    t_glob = "will follow automatically from its "

    res = normalize(s_glob, t_glob, l, n)
    print(res)



cProfile.run('main()')
