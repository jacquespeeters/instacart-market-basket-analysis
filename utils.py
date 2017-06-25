import pandas as pd

# Thanks to https://www.kaggle.com/onodera/multilabel-fscore
def multilabel_fscore(y_true, y_pred):
    """
    ex1:
    y_true = [1, 2, 3]
    y_pred = [2, 3]
    return: 0.8

    ex2:
    y_true = ["None"]
    y_pred = [2, "None"]
    return: 0.666

    ex3:
    y_true = [4, 5, 6, 7]
    y_pred = [2, 4, 8, 9]
    return: 0.25

    """
    precision = sum([1 for i in y_pred if i in y_true]) / len(y_pred)
    recall = sum([1 for i in y_true if i in y_pred]) / len(y_true)
    denom = (precision + recall)
    # ipdb.set_trace()
    if denom == 0:
        denom = 1
    return (2 * precision * recall) / denom


def add_none_to_order_set(df):
    # If atleast one reordered
    df_none = df. \
        groupby(["order_id", "user_id"])["reordered"].sum().\
        reset_index()

    # Then None to O
    df_none["reordered"] = (df_none["reordered"] < 1).astype(int)

    # product_id for None is defined as 0
    df_none["product_id"] = 0
    df_none["add_to_cart_order"] = np.nan
    # reorder columns
    df_none = df_none[df.columns]
    df = pd.concat([df, df_none])
    return df


# Useless now
def add_none_to_order_set(df):
    # If atleast one reordered
    df_none = df. \
        groupby(["order_id", "user_id"])["reordered"].sum().\
        reset_index()

    # Then None to O
    df_none["reordered"] = (df_none["reordered"] < 1).astype(int)

    # product_id for None is defined as 0
    df_none["product_id"] = 0
    df_none["add_to_cart_order"] = np.nan
    # reorder columns
    df_none = df_none[df.columns]
    df = pd.concat([df, df_none])
    return df

#order_products__prior = add_none_to_order_set(order_products__prior)
#order_products__train = add_none_to_order_set(order_products__train)
