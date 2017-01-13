# -*- coding: utf-8 -*-
import argparse
from sklearn import svm
import numpy as np

from reuters import *
from utils import get_table_values
from bgm import build_gram_matrix as bgm
from gram_for_data import write_gram_to_file, read_gram_from_file
from approx import get_K, get_S, get_top_S


"""
Classification for bag of words and ngram.
Use -d or --download for the first time to load all the data.

Example use:
  python3 classify.py bow -i 10
  python3 classify.py ngram -n 3 -i 10
  python3 classify.py ssk -n 2 -l 0.5 -i 1

Author: Þorsteinn Daði Gunnarsson
"""


def train_and_test_classifier(datasets, method, n=None, l=None):
    ys, pr = [], []

    for data in datasets:
        train = data["train"]
        test = data["test"]
 
        if method == "ssk":
            clf = svm.SVC(kernel="precomputed")
            try:
                d = read_gram_from_file(100, n, l)
                X = d["X"]
                Y = d["Y"]
            except IOError:
                S = get_top_S(train["x"], n, count=10)
                K = get_K(S)
                x = train["x"]
                X = bgm(x, x, l, n, K=K)
                y = test["x"]
                Y = bgm(y, x, l, n, K=K)
                write_gram_to_file(0, n, l, X, Y)
        else: 
            if method == "bow":
                vectorizer, X = get_bow(train["x"])
            elif method == "ngram":
                vectorizer, X = get_ngram(train["x"], n)

            clf = svm.LinearSVC(multi_class="ovr")
#            clf = svm.SVC(decision_function_shape="ovr")
            Y = normalize(vectorizer.transform(test["y"]).toarray())
        clf.fit(X, train["y"])  # Train classifier

        ys.append(test["y"])
        pr.append(clf.predict(Y))

    stats = get_table_values(datasets[0]["categories"], ys, pr)
    keys = ["F1", "precision", "recall"]
    print("{:10} {:20} {:20} {:20}".format(*(["Category"] + keys)))
    for cat in data["categories"]:
        values = stats[cat]
        print("{:10} {:20} {:20} {:20}".format(*([cat] + ["{:.3f} ({:.3f})".format(*values[k]) for k in keys])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Classify categories from Reuters dataset with a SVM")
    parser.add_argument("Vectorizer", help="What method to use to vectorize strings ", choices=["bow", "ngram", "ssk"])
    parser.add_argument("-n", type=int, help="N", default=2)
    parser.add_argument("-l", type=float, help="l", default=0.5)
    parser.add_argument("-i", type=int, help="Dataset", default=1)
    parser.add_argument("-d", "--download", action="store_true", help="Downloads all data needed")
    res = vars(parser.parse_args())

    i = res["i"]
    method = res["Vectorizer"]
    n = res["n"]
    l = res["l"]

    if method == "bow":
        method_name = "bag of words"
    elif method == "ngram":
        method_name = "ngram (n={})".format(n)
    elif method == "ssk":
        method_name = "ssk (n={}, l={})".format(n, l)

    print("Classifying i={} datasets using {}".format(i, method_name)) 

    if res["download"]:
        download()

    from simple_data import DATA as DATA
    train_and_test_classifier(DATA[:i], method, n=n, l=l)

