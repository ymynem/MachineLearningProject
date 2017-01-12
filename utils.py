import json
from numpy import mean, std  # Its not what you think
from sklearn.metrics import f1_score as f1, precision_score as precision, recall_score as recall


def write_data(name):
    """
    Write data object to file
    """
    pass


def load_data(name):
    """
    Load data object from file
    """
    pass


def get_table_values(y_true, y_predicted):
    zipped = list(zip(y_true, y_predicted))
    f1s = [f1(y_t, y_p, average=None) for y_t, y_p in zipped]
    pres = [precision(y_t, y_p, average=None) for y_t, y_p in zipped]
    recs = [recall(y_t, y_p, average=None) for y_t, y_p in zipped]
    values = {
        "F1": (mean(f1s), std(f1s)),
        "precision": (mean(pres), std(pres)),
        "recall": (mean(recs), std(recs)),
    }
    return values

