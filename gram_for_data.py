import json
import argparse

from text_classifier import buildGramMat as bgm
from simple_data import SIMPLE_DATA


def get_gram_matrices(x, y, n, l):
    print("starting x, x", len(x))
    X = bgm(x, x, l, n)
    print("starting x, y", len(x), len(y))
    Y = bgm(x, y, l, n)
    return X, Y


def write_gram_to_file(i, n, l, X, Y):
    data = {"i": i, "n": n, "l": l, "X": X, "Y": Y}
    with open("grams/gram-{}-n{}-l{}.json", "w") as outfile:
        json.dump(data, outfile)


def save_grams_to_file(n, l):
    for i, d in zip(range(len(SIMPLE_DATA)), SIMPLE_DATA):
        X, Y = get_gram_matrices(d["train"]["x"], d["test"]["x"], n, l)
        write_gram_to_file(i, n, l, X, Y)
        print(i, "Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute and save gram matrices for SIMPLE_DATA")
    parser.add_argument("n", type=int)
    parser.add_argument("l", type=float)

    res = parser.parse_args()
    save_grams_to_file(res.n, res.l)

