import pandas as pd
import numpy as np

# Thanks to https://www.kaggle.com/onodera/multilabel-fscore
def multilabel_fscore(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array([y_pred])

    precision = sum([1 for i in y_pred if i in y_true]) / len(y_pred)
    recall = sum([1 for i in y_true if i in y_pred]) / len(y_true)
    denom = (precision + recall)
    # ipdb.set_trace()
    if denom == 0:
        denom = 1
    return (2 * precision * recall) / denom


