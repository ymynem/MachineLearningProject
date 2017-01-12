from reuters import *
from random import Random
from collections import defaultdict


def get_data(categories, seed=0):
    cats = {}
    data = defaultdict(lambda: {"x": [], "y": []})
    data["categories"] = []
    for key, value in categories.items():
        data["categories"].append(key)
        train, test = get_documents(key)
        Random(seed).shuffle(train)
        Random(seed).shuffle(test)
        cats[key] = [create_corpus(train[:value[0]]), create_corpus(test[:value[1]])]
    for key, value in cats.items():
        data["train"]["x"].extend(value[0])
        data["train"]["y"].extend([key]*len(value[0]))
        data["test"]["x"].extend(value[1])
        data["test"]["y"].extend([key]*len(value[1]))
    return data


def get_bulks(n, COUNTS):
    bulks = []
    for i in range(n):
        bulks.append(get_data(COUNTS, seed=i+10))

#    print(bulks[0]["train"]["x"][0].split(" ")[0])
    return bulks


SIMPLE_DATA_BULK_COUNT = 10
SIMPLE_DATA = get_bulks(SIMPLE_DATA_BULK_COUNT, {
        "earn" : (152, 40),
        "acq"  : (114, 25),
        "crude": (76, 15),
        "corn" : (38, 10),
    })
