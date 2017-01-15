import json
from numpy import mean, std  # Its not what you think
from sklearn.metrics import f1_score as f1, precision_score as precision, recall_score as recall


def write_data(name, data):
    """
    Write data object to file
    """
    with open(name, "w") as of:
        json.dump(data, of)


def load_data(name):
    """
    Load data object from file
    """
    with open(name, "r") as of:
        data = json.load(of)
    return data


def get_table_values(cats, y_true, y_predicted):
    zipped = list(zip(y_true, y_predicted))
    f1s = [f1(y_t, y_p, average=None) for y_t, y_p in zipped]
    pres = [precision(y_t, y_p, average=None) for y_t, y_p in zipped]
    recs = [recall(y_t, y_p, average=None) for y_t, y_p in zipped]
    values = {}
    for i, cat in zip(range(len(cats)), cats):
        values[cat] = {
            "F1": (mean([v[i] for v in f1s]), std([v[i] for v in f1s])),
            "precision": (mean([v[i] for v in pres]), std([v[i] for v in pres])),
            "recall": (mean([v[i] for v in recs]), std([v[i] for v in recs])),
        }
    return values

