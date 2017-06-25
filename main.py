import pandas as pd
import numpy as np
import lightgbm as lgb
import utils
import ipdb
import timeit

# Thanks for the inspiration
# https://www.kaggle.com/paulantoine/light-gbm-benchmark-0-3692/code

# Read data -----------------------------
# TODO function to read csv
aisles = pd.read_csv("./data/aisles.csv")
departments = pd.read_csv("./data/departments.csv")
order_products__prior = pd.read_csv("./data/order_products__prior.csv")
order_products__train = pd.read_csv("./data/order_products__train.csv")
orders = pd.read_csv("./data/orders.csv")
products = pd.read_csv("./data/products.csv")

# Data wrangling ----------------------------------------------------

print("Sample by user_id")
user_id = orders["user_id"].unique()
np.random.seed(seed=7)
sample_user_id = np.random.choice(user_id, size=1000, replace=False).tolist()
orders = orders.query("user_id == @sample_user_id")

###

# Fucking slow :/
def add_fe_to_orders(group):
    group["date"] = group.ix[::-1, 'days_since_prior_order'].cumsum()[::-1].shift(-1).fillna(0)
    max_group = group["order_number"].max()
    group["order_number_reverse"] = max_group - group["order_number"]
    return group

orders = orders.groupby("user_id").\
    apply(add_fe_to_orders)

# TODO : redundancy of datasets here, should make it cleaner
print("Add user_id to order_products__XXX ")
order_products__prior = pd.merge(orders[["order_id", "user_id"]], order_products__prior, on="order_id")
order_products__train = pd.merge(orders[["order_id", "user_id"]], order_products__train, on="order_id")

order_prior = pd.merge(orders, order_products__prior, on=["order_id", "user_id"])
order_train = pd.merge(orders, order_products__train, on=["order_id", "user_id"])

### products_fe - feature engineering on products
print("feature engineering on products")
products_fe = pd.DataFrame()

products_fe["orders"] = order_products__prior.\
    groupby("product_id").\
    size()

products_fe["reorders"] = order_products__prior.\
    query("reordered == 1").\
    groupby("product_id").\
    size()
# Set null size to 0
products_fe["reorders"][products_fe["reorders"].isnull()] = 0

# TODO - from UP get reorder_ratio in order_number and days of each product which is equivalent to frequency

# TODO : haha l'indicateur est pas bon du tout ici :D
products_fe['reorder_rate'] = products_fe["reorders"] / products_fe["orders"]

products_fe["add_to_cart_order"] = order_products__prior.\
    groupby("product_id")["add_to_cart_order"].\
    mean()

products_fe = products_fe.reset_index()

### user_fe - Feature engineering on user
print("Feature engineering on user")

# TODO : warning here, should solve this
# TODO : can be cleaner?
# days_since_prior_order info
users_fe = orders.\
    groupby("user_id")["days_since_prior_order"].\
    agg([np.sum, np.mean, np.std])

users_fe.columns = ["sum_days_since_prior_order", "mean_days_since_prior_order", "std_days_since_prior_order"]
users_fe = users_fe.reset_index()

# User basket sum, mean, std
users_fe2 = order_products__prior.\
    groupby(["user_id", "order_id"]).size().\
    reset_index().\
    drop("order_id", axis=1).\
    groupby("user_id").\
    agg([np.sum, np.mean, np.std])

users_fe2.columns = ["sum_basket", "mean_basket", "std_basket"]
users_fe2 = users_fe2.reset_index()

users_fe3 = order_products__prior.\
    groupby("user_id")["reordered"].\
    agg([np.mean]).\
    reset_index()

users_fe3.columns = ["user_id", "U_rt_reordered"]

users_fe = pd.merge(users_fe, users_fe2, on="user_id")
users_fe = pd.merge(users_fe, users_fe3, on="user_id")
del users_fe2, users_fe3

### user_product - UP
print("Feature engineering on User_product")

# Could be something else than 1/2
order_prior["UP_date_strike"] = 1/2 ** (order_prior["date"]/7)
order_prior["UP_order_strike"] = 1/2 ** (order_prior["order_number_reverse"])
users_products = order_prior.\
    groupby(["user_id", "product_id"]).\
    agg({'reordered': {'nb_reordered': "size"} ,\
         'add_to_cart_order': {'mean_add_to_cart_order': "mean"},\
         'order_number': {'last_order_number': "max", 'first_order_number': "min"}, \
         'date': {'last_order_date': "min", 'first_date_number': "max"}, \
         'UP_date_strike': {"UP_date_strike": "sum"},\
         'UP_order_strike': {"UP_order_strike": "sum"}})

users_products.columns = users_products.columns.droplevel(0)
users_products = users_products.reset_index()
users_products.head()

### Construct target Y dataset by creating user_past_product
print("Get label by creating user_past_product")
# Get for each user all the product he has already bought before
user_past_product = order_products__prior.\
    groupby("user_id")["product_id"].unique().\
    reset_index()

# Reshaping => equivalent to explode function
user_past_product = user_past_product.set_index('user_id')['product_id'].\
    apply(pd.Series).stack().reset_index()

user_past_product = user_past_product.drop("level_1", axis=1)
user_past_product.columns = ["user_id", "product_id"]
user_past_product["product_id"] = user_past_product["product_id"].astype(int)

user_past_product.head()

reordered_train = pd.merge(orders, order_products__train, on=["order_id", "user_id"])
reordered_train = reordered_train.query("reordered == 1")
reordered_train.head()

user_past_product = pd.merge(user_past_product, reordered_train[["user_id", "product_id", "reordered"]],
         on=["user_id", "product_id"], how="left")

user_past_product["reordered"] = user_past_product["reordered"].fillna(0)

del reordered_train


# Create dataset which we'll learn on -------------------------------------------------

def get_df(df):
    df_set = pd.merge(df, user_past_product, on=["user_id"])
    df_set = pd.merge(df_set, users_fe, on="user_id")
    df_set = pd.merge(df_set, products_fe, how='left', on="product_id")
    df_set = pd.merge(df_set, products, how='left', on="product_id")
    df_set = pd.merge(df_set, aisles, how='left', on="aisle_id")
    df_set = pd.merge(df_set, departments, how='left', on="department_id")
    df_set = pd.merge(df_set, users_products, how='left', on=["user_id", "product_id"])

    # Should be done in appropriate place
    df_set["up_rt_reordered"] =  df_set["nb_reordered"] / df_set["order_number"]
    df_set["up_nb_order_since_first"] = (df_set["order_number"] - df_set["first_order_number"])
    df_set["up_rt_reordered_since_first"] = df_set["nb_reordered"] / df_set["up_nb_order_since_first"]
    df_set["up_nb_no-reordered_since_last"] = df_set["order_number"] - df_set["last_order_number"]

    return df_set

df_train = get_df(orders.query("eval_set == 'train'"))
df_test = get_df(orders.query("eval_set == 'test'"))

# Modeling -----------------------
# TODO: sample by user
print("Sample by user_id")
user_id = df_train["user_id"].unique()
np.random.seed(seed=7)
sample_user_id = np.random.choice(user_id, size=int(len(user_id)*0.2), replace=False).tolist()
sample_index = df_train.query("user_id == @sample_user_id").index

to_drop = ["order_id", "user_id", "eval_set", "product_id", "product_name","department", "aisle", \
           "order_number_reverse"]

X_train = df_train.drop(to_drop + ["reordered"], axis=1)
X_test = df_test.drop(to_drop + ["reordered"], axis=1)
y_train = df_train["reordered"]

X_valid = X_train.ix[sample_index]
y_valid = y_train.ix[sample_index]
X_train = X_train.drop(sample_index)
y_train = y_train.drop(sample_index)

# TODO replace by default lightgbm app and not the sklearn API
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=200000)

model_gbm = gbm.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric='l1',
        early_stopping_rounds=50)

# Predict and submit -------------------------------------------------------------
print('Predict and submit')

def get_df_pred(dataset="test"):
    # Probably not smart
    if dataset=="test":
        df = df_test
        X_df = X_test
    else:
        df = df_train.ix[sample_index]
        X_df = X_valid

    df["pred"] = model_gbm.predict(X_df)
    df["pred"] = np.clip(df['pred'], 0, 1)  # Lightgbm predict values <0 and >1

    # None is the complementary of reordering at least one product
    # Copy avoid warning
    df_pred = df[["order_id", "user_id", "product_id", "pred"]].copy()

    df_pred["pred_minus"] = 1-df_pred["pred"]
    df_pred_none = df_pred.groupby(["order_id", "user_id"])["pred_minus"].agg("prod").rename("pred").reset_index()
    df_pred = df_pred.drop("pred_minus", axis=1)

    df_pred_none["product_id"] = "None"
    df_test_pred = pd.concat([df_pred, df_pred_none])
    return df_test_pred

df_test_pred = get_df_pred("test")
df_valid_pred = get_df_pred("valid")

def groupby_optimised_pred(group):
    #ipdb.set_trace()
    group_none = group.query("product_id == 'None'")
    f_score_none = group_none["pred"].values[0]

    group_no_none = group.query("product_id != 'None'")
    group_no_none = group_no_none.sort_values(["pred"], ascending=False)
    group_no_none["precision"] = group_no_none["pred"].expanding().mean()
    group_no_none["recall"] = group_no_none["pred"].expanding().sum() / sum(group_no_none["pred"])
    group_no_none["f_score"] =  (2 * group_no_none["precision"] * group_no_none["recall"]) / (group_no_none["precision"] + group_no_none["recall"])
    #group["rank"] = group["pred"].rank(ascending = False)
    f_score_no_none = max(group_no_none["f_score"])
    max_index = np.where(group_no_none["f_score"] == f_score_no_none)[0][0]
    group_no_none = group_no_none[0:max_index].drop(["precision", "recall", "f_score"], axis=1)

    #ipdb.set_trace()
    if f_score_none > f_score_no_none:
        res = group_none
    else :
        res = group_no_none

    return res


def filter_optimised_pred(df):
    df = df. \
        groupby("user_id"). \
        apply(groupby_optimised_pred).reset_index(drop=True)
    return df


def compute_fscore(df_valid_pred):
    df_valid = df_train.ix[sample_index]

    df_none_true = df_valid. \
        groupby(["order_id", "user_id"])["reordered"].sum().\
        reset_index()

    # If atleast one reorderd then None is 0, otherwise 1
    df_none_true["reordered"] = (df_none_true["reordered"] < 1).astype(int)
    df_none_true = df_none_true.query("reordered == 1")
    # product_id for None is defined as 0
    df_none_true.loc[:,"product_id"] = "None"
    df_valid = pd.concat([df_valid, df_none_true])
    df_y_true = df_valid.query("reordered==1").groupby("user_id")["product_id"].unique().\
    reset_index()
    df_y_true.columns = ["user_id", "y_true"]
    df_y_pred = df_valid_pred.groupby("user_id")["product_id"].unique().\
    reset_index()
    df_y_pred.columns = ["user_id", "y_pred"]

    # Should be not be an inner join
    res = pd.merge(df_y_true, df_y_pred, on="user_id")

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
        #ipdb.set_trace()
        if denom == 0:
            denom = 1
        return (2 * precision * recall) / denom

    res = res.apply(lambda x: multilabel_fscore(x["y_true"], x["y_pred"]), axis=1)
    return res.mean()

print("Score estimation")
print(compute_fscore(df_valid_pred.query("pred > 0.22")))
# Doesn't work because basket size of reordered product is probably not good. And None product is something really particular
#print(compute_fscore(filter_optimised_pred(df_valid_pred)))

TRESHOLD = 0.22 # guess, should be tuned with crossval on a subset of train data
d = dict()

df_test_pred["product_id"].head()

for row in df_test_pred.itertuples():
    if row.pred > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in df_test_pred.order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('./sub.csv', index=False)



