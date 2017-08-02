import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# Thanks to https://www.kaggle.com/onodera/multilabel-fscore
def multilabel_fscore(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array([y_pred])

    precision = sum([1 for i in y_pred if i in y_true]) / len(y_pred)
    recall = sum([1 for i in y_true if i in y_pred]) / len(y_true)
    denom = (precision + recall)
    if denom == 0:
        denom = 1
    return (2 * precision * recall) / denom


def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)
