from reuters import *
from random import Random
from collections import defaultdict, OrderedDict


def get_data(categories, seed=0):
    cats = OrderedDict()
    data = defaultdict(lambda: {"x": [], "y": []})
    data["categories"] = []
    for key, value in categories:
        data["categories"].append(key)
        train, test = get_documents(key)
        Random(seed).shuffle(train)
        Random(seed).shuffle(test)
        MAX_LENGTH = 100
        cats[key] = [
            create_corpus(train[:value[0]], max_length=MAX_LENGTH), 
            create_corpus(test[:value[1]], max_length=MAX_LENGTH)]
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
SIMPLE_DATA = get_bulks(SIMPLE_DATA_BULK_COUNT, [
        ("earn", (152, 40)),
        ("acq", (114, 25)),
        ("crude", (76, 15)),
        ("corn", (38, 10)),
    ])


VERY_SIMPLE_DATA = get_bulks(10, [ 
        ("earn", (50, 15)),
        ("acq", (50, 15)),
        ("crude", (50, 15)),
        ("corn", (38, 10)),
    ])
       
DATA = get_bulks(10, [
        ("earn", (25, 8)),
        ("acq", (25, 8)),
        ("crude", (25, 8)),
        ("corn", (25, 8)),
    ])

