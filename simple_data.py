from reuters import *
from random import shuffle, seed
from collections import defaultdict


seed(0)  # So everybody gets the same bulks


def get_data(categories):
    cats = {}
    for key, value in categories.items():
        train, test = get_documents(key)
        shuffle(train)
        shuffle(test)
        cats[key] = [create_corpus(train[:value[0]]), create_corpus(test[:value[1]])]
    data = defaultdict(lambda: {"x": [], "y": []})
    for key, value in cats.items():
        data["train"]["x"].extend(value[0])
        data["train"]["y"].extend([key]*len(value[0]))
        data["test"]["x"].extend(value[1])
        data["test"]["y"].extend([key]*len(value[1]))
    return data


SIMPLE_DATA_BULK_COUNT = 10
SIMPLE_DATA = [get_data({
        "earn" : (152, 40),
        "acq"  : (114, 25),
        "crude": (76, 15),
        "corn" : (38, 10),
    }) for i in range(SIMPLE_DATA_BULK_COUNT)]
