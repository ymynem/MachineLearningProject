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
            s_counter_minus_one = s_counter - 1
            x = s[s_counter_minus_one]  # last character. sx means the hole string, when they write only s they mean exclude last char
            the_sum = 0
            i_minus_one = i -1
            for j in range(jtot):
                if x == t[j]:
                    the_sum += subsequence_kernel_primed(s_counter_minus_one, j, l, i_minus_one) * l ** (jtot - j + 2)
        res = l * subsequence_kernel_primed(s_counter_minus_one, jtot, l, i) + the_sum
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
    lru_wrapped_func = subsequence_kernel_primed_lru_wrapped(s, t)
    if min(s_len, len(t)) < n:
        return 0
    else:
        the_sum = 0
        if n > 0:
            x = s[-1]
            s_len_minus_one = s_len - 1
            n_minus_one = n - 1
            for j in range(len(t)):
                if t[j] == x:
                    the_sum += lru_wrapped_func(s_len_minus_one, j, l, n_minus_one) * l ** 2  # [:-1]
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

    """
    s= "science is organized knowledge"
    t= "wisdom is organized life"
     K1 = 0.580, K2 = 0.580, K3 = 0.478, K4 = 0.439, K5 = 0.406, K6 = 0.370
    """
    #   print("1: ", subsequence_kernel_primed("car", "cat", 0.5, 2))
    #   s = "Anyone who reads Old and Middle English"
    #   t = "Πόθεν, ὦ Σώκρατες, φαίνῃ; Ἤ δῆλα δὴ ὅτι"
    l = 0.5
    n = 2
    s = "wisdom is organized life is franca.Named after the Angles, wisdom is organized life one of the Germanic "
    t = "English is a West Germanic language that was first spoken in early medieval England and is franca.Named after the Angles, "
    print("s = ",len(s))
    print("t = ",len(t))

    if s == t:
        res = 1
    else:
        res = normalize(s, t, l, n)
    print(res)



cProfile.run('main()')
