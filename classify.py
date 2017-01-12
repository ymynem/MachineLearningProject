# -*- coding: utf-8 -*-
import argparse
from sklearn import svm

from reuters import *
from utils import get_table_values


"""
Classification for bag of words and ngram.
Use -d or --download for the first time to load all the data.

Example use:
  python3 classify.py acq corn bow
  python3 classify.py acq corn ngram -n 3


Author: Þorsteinn Daði Gunnarsson
"""


def train_and_test_classifier(datasets, method, n=2):
    ys, pr = [], []

    for data in datasets:
        train = data["train"]
        test = data["test"]
  
        if method == "bow":
            vectorizer, X = get_bow(train["x"])
        elif method == "ngram":
            vectorizer, X = get_ngram(train["x"], n)

        clf = svm.SVC(decision_function_shape="ovr")
        clf.fit(X, train["y"])  # Train classifier

        ys.append(test["y"])
        pr.append(clf.predict(vectorizer.transform(test["x"]).toarray()))

    stats = get_table_values(data["categories"], ys, pr)
    keys = ["F1", "precision", "recall"]
    print("{:10} {:20} {:20} {:20}".format(*(["Category"] + keys)))
    for cat in data["categories"]:
        values = stats[cat]
        print("{:10} {:20} {:20} {:20}".format(*([cat] + ["{:.3f} ({:.3f})".format(*values[k]) for k in keys])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Classify categories from Reuters dataset with a SVM")
    parser.add_argument("Vectorizer", help="What method to use to vectorize strings ", choices=["bow", "ngram"])
    parser.add_argument("-n", type=int, help="N", default=2)
    parser.add_argument("-i", type=int, help="Dataset", default=1)
    parser.add_argument("-d", "--download", action="store_true", help="Downloads all data needed")
    res = vars(parser.parse_args())

    i = res["i"]
    method = res["Vectorizer"]
    n = res["n"]

    if method == "bow":
        method_name = "bag of words"
    elif method == "ngram":
        method_name = "ngram (n={})".format(n)

    print("Classifying dataset {} using {}".format(i, method_name)) 

    if res["download"]:
        download()

    from simple_data import SIMPLE_DATA
    train_and_test_classifier(SIMPLE_DATA[:i], method, n=n)

