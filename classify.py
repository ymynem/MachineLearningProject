import argparse
from sklearn import svm

from reuters import *


"""
Classification for bag of words and ngram.
Use -d or --download for the first time to load all the data.

Example use:
  python3 classify.py acq corn bow
  python3 classify.py acq corn ngram -n 3


Author: Þorsteinn Daði Gunnarsson
"""


def train_and_test_classifier(cat1, cat2, method, n=2):
    # Get document ids
    a_train, a_test = get_documents(cat1)
    b_train, b_test = get_documents(cat2)

    print("Number of documents:", len(a_train), len(b_train))

    train_data = create_corpus(a_train + b_train)
    y = [cat1]*len(a_train) + [cat2]*len(b_train)

    if method == "bow":
        vectorizer, X = get_bow(train_data)
    elif method == "ngram":
        vectorizer, X = get_ngram(train_data, n)

    clf = svm.SVC()
    clf.fit(X, y)  # Train classifier

    print("Number of support vectors:", clf.n_support_)

    test_corpus = create_corpus(a_test + b_test)

    X_test = vectorizer.transform(test_corpus).toarray()
    y_test = [cat1]*len(a_test) + [cat2]*len(b_test)

    pr = clf.predict(X_test)
    total_correct = len([1 for p in zip(pr, y_test) if p[0] == p[1]])
    print("Results: {}/{} - {:.2f}%".format(total_correct, len(pr), 100*total_correct/len(pr)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Classify categories from Reuters dataset with a SVM")
    parser.add_argument("Category 1", help="The first categury to classify")
    parser.add_argument("Category 2", help="The second categury to classify")
    parser.add_argument("Vectorizer", help="What method to use to vectorize strings ", choices=["bow", "ngram"])
    parser.add_argument("-n", type=int, help="N", default=2)
    parser.add_argument("-d", "--download", action="store_true", help="Downloads all data needed")
    res = vars(parser.parse_args())

    cat1 = res["Category 1"]
    cat2 = res["Category 2"]
    method = res["Vectorizer"]
    n = res["n"]

    if method == "bow":
        method_name = "bag of words"
    elif method == "ngram":
        method_name = "ngram (n={})".format(n)

    print("Classifying {} and {} using {}".format(cat1, cat2, method_name)) 

    if res["download"]:
        download()
    train_and_test_classifier(cat1, cat2, method, n=n)

