import pandas as pd
import numpy as np
import lightgbm as lgb
import utils
import ipdb
import timeit
import pickle
import gc
import time
from multiprocessing import Pool, cpu_count

# Thanks for the inspiration
# https://www.kaggle.com/paulantoine/light-gbm-benchmark-0-3692/code

# Read data -----------------------------
# TODO function to read csv
aisles = pd.read_csv("./data/aisles.csv")
departments = pd.read_csv("./data/departments.csv")
order_prior = pd.read_csv("./data/order_products__prior.csv")
order_train = pd.read_csv("./data/order_products__train.csv")
orders = pd.read_csv("./data/orders.csv")
products = pd.read_csv("./data/products.csv")

# Fucking slow :/ - So i pickled it
# Way faster & easier with dplyr...
def add_fe_to_orders(group):
    group["date"] = group.ix[::-1, 'days_since_prior_order'].cumsum()[::-1].shift(-1).fillna(0)
    max_group = group["order_number"].max()
    group["order_number_reverse"] = max_group - group["order_number"]
    return group

#orders = orders.groupby("user_id").\
#    apply(add_fe_to_orders)

#pickle.dump(orders, open("orders.p", "wb"))
orders = pickle.load(open("orders.p", "rb"))
product2vec = pickle.load(open("product2vec.p", "rb"))

# Data wrangling ----------------------------------------------------

print("Sample by user_id")
user_id = orders["user_id"].unique()
np.random.seed(seed=7)
sample_user_id = np.random.choice(user_id, size=20000, replace=False).tolist()
#orders = orders.query("user_id == @sample_user_id")

###
print("Add user_id to order_products__XXX ")

order_prior = pd.merge(orders, order_prior, on=["order_id"])
order_train = pd.merge(orders, order_train, on=["order_id"])

# Add last_basket_size
#last_basket_size = order_prior.groupby(["user_id", "order_number_reverse"]).size().\
#    rename("last_basket_size").reset_index()
#last_basket_size["order_number_reverse"] = last_basket_size["order_number_reverse"] - 1

#orders = pd.merge(orders, last_basket_size, how="left", on=["user_id", "order_number_reverse"])

#del last_basket_size

### products_fe - Feature engineering on products
print("Feature engineering on products")

products_fe = order_prior.\
    groupby(["product_id"]).\
    agg({'reordered': {'p_reorder_rt': "mean", 'p_count': "size"},\
         'add_to_cart_order': {"p_add_to_cart_order": "mean"},\
         'order_number': {"P_recency_order": "mean"}, \
         'order_number_reverse': {"P_recency_order_r": "mean"}, \
         'date': {"P_recency_date": "mean"}})

products_fe.columns = products_fe.columns.droplevel(0)
products_fe = products_fe.reset_index()

products_fe.head()

# bool_reordered = if a product is bought once, prob to be reordered at least once
up_fe2 = order_prior.groupby(["user_id", "product_id"]).agg("size").rename("up_nb_ordered").reset_index()
up_fe2["bool_reordered"] = (up_fe2["up_nb_ordered"] > 1).astype("int")

products_fe2 = up_fe2.groupby('product_id')["bool_reordered"].\
    agg(["mean","size"]).reset_index().\
    rename(index=str, columns={"mean": "p_reorder_rt_bool", "size": "p_active_user"})

products_fe = pd.merge(products_fe, products_fe2, how="left", on="product_id")
del products_fe2

# Product trend in a way
products_trend = order_prior.query("order_number_reverse < 3").\
    groupby(["product_id", "order_number_reverse"]).size().\
    rename("p_size").reset_index()

products_trend["p_trend_rt"] = products_trend["p_size"] / products_trend["p_size"].shift(-1)
products_trend["p_trend_diff"] = products_trend["p_size"] - products_trend["p_size"].shift(-1)

cond = products_trend["product_id"] != products_trend["product_id"].shift(-1)
products_trend.loc[cond, "p_trend_rt"] = np.nan
products_trend.loc[cond, "p_trend_diff"] = np.nan
products_trend = products_trend.query("order_number_reverse == 1").drop("order_number_reverse", 1)

products_fe = pd.merge(products_fe, products_trend, how="left", on="product_id")

del cond, products_trend

product_freq = order_prior.copy()
product_freq = product_freq.sort_values(["user_id", "product_id", "order_number"])

product_freq["p_freq_days"] = product_freq["date"].shift() - product_freq["date"]
product_freq["p_freq_order"] = product_freq["order_number"] - product_freq["order_number"].shift()
product_freq = product_freq.query("reordered == 1")

product_freq = product_freq.groupby("product_id").\
    agg({'p_freq_days': {'p_freq_days': "mean"}, \
         'p_freq_order': {'p_freq_order': "mean"}})

product_freq.columns = product_freq.columns.droplevel(0)
product_freq = product_freq.reset_index()

products_fe = pd.merge(products_fe, product_freq, how="left", on="product_id")

del product_freq

# Doesn't seem to help
products_organic = products[["product_id", "product_name"]].copy()
products_organic["organic_bool"] = products["product_name"].str.match("organic", case=False).astype('int')
products_fe = pd.merge(products_fe, products_organic.drop("product_name", axis=1), how="left", on="product_id")
del products_organic

### user_fe - Feature engineering on user
print("Feature engineering on user")
users_fe = order_prior.\
    groupby("user_id").\
    agg({'reordered':{'U_rt_reordered':'mean'},\
         'date':{'U_date_inscription':'max'},\
         'days_since_prior_order': {'U_days_since_mean': 'mean', \
                                   'U_days_since_std': 'std', \
                                   'U_days_since_sum': 'sum'}})

users_fe.columns = users_fe.columns.droplevel(0)
users_fe = users_fe.reset_index()

# User basket sum, mean, std
# TODO same but only on reordered products?
users_fe2 = order_prior.\
    groupby(["user_id", "order_id"]).size().\
    reset_index().\
    drop("order_id", axis=1).\
    groupby("user_id").\
    agg([np.sum, np.mean, np.std])

users_fe2.columns = ["U_basket_sum", "U_basket_mean", "U_basket_std"]
users_fe2 = users_fe2.reset_index()

users_fe = pd.merge(users_fe, users_fe2, on="user_id")
del users_fe2

# u_active_p == user distinct products
users_fe4 = up_fe2.groupby('user_id')["bool_reordered"].\
    agg(["mean","size"]).reset_index().\
    rename(index=str, columns={"mean": "u_reorder_rt_bool", "size": "u_active_p"})

users_fe = pd.merge(users_fe, users_fe4, on="user_id")
del users_fe4

# TODO U_none_reordered_strike
# New way
# TODO test if it's help or not, keeping it might be used as a skrinking term
# .query("order_number !=1")
user_fe_none = order_prior.\
  groupby(["order_id", "user_id"]).\
  agg({'reordered': {'reordered': "sum"},\
       'order_number_reverse': {'order_number_reverse':'first'}})

user_fe_none.columns = user_fe_none.columns.droplevel(0)
user_fe_none = user_fe_none.reset_index()

user_fe_none["reordered"] = (user_fe_none["reordered"] < 1).astype(int)
user_fe_none["U_none_reordered_strike"] = user_fe_none["reordered"] * 1/2 ** (user_fe_none["order_number_reverse"])

user_fe_none = user_fe_none.\
    groupby("user_id"). \
    agg({'reordered': {'U_none_reordered_mean': "mean", 'U_none_reordered_sum': "sum"}, \
         'U_none_reordered_strike': {'U_none_reordered_strike': "sum"}})

user_fe_none.columns = user_fe_none.columns.droplevel(0)
user_fe_none = user_fe_none.reset_index()

users_fe = pd.merge(users_fe, user_fe_none, on="user_id")
del user_fe_none

### user_product - UP
print("Feature engineering on User_product")
# Could be something else than 1/2
order_prior["UP_date_strike"] = 1/2 ** (order_prior["date"]/7)
#order_prior["UP_order_strike"] = 100000 * 1/2 ** (order_prior["order_number_reverse"])
order_prior["UP_order_strike"] = 1/2 ** (order_prior["order_number_reverse"])

users_products = order_prior.\
    groupby(["user_id", "product_id"]).\
    agg({'reordered': {'up_nb_reordered': "size"},\
         'add_to_cart_order': {'up_mean_add_to_cart_order': "mean"},\
         'order_number_reverse': {'up_last_order_number': "min", 'up_first_order_number': "max"}, \
         'date': {'up_last_order_date': "min", 'up_first_date_number': "max"}, \
         'UP_date_strike': {"UP_date_strike": "sum"},\
         'UP_order_strike': {"UP_order_strike": "sum"}})

users_products.columns = users_products.columns.droplevel(0)
users_products = users_products.reset_index()

#users_products["UP_order_strike_rt"] = users_products["UP_order_strike"] / ((1 - 1/2**(users_products["up_first_order_number"] + 1))/(1-1/2) - 1)

### users_products_none summary of users_products

users_products_none = users_products.groupby("user_id").\
    agg({'UP_date_strike' : {'O_date_strike_max' : "max", 'O_date_strike_sum' : "sum", 'O_date_strike_mean' : "mean"}, \
         'UP_order_strike': {'O_order_strike_max': "max", 'O_order_strike_sum': "sum", 'O_date_order_mean': "mean"}})

users_products_none.columns = users_products_none.columns.droplevel(0)
users_products_none = users_products_none.reset_index()

### aisles
print("Feature engineering on aisles")
aisles_order = pd.merge(order_prior, products, on="product_id")
aisles_order = pd.merge(aisles_order, aisles, on="aisle_id")

aisles_fe = aisles_order.\
    groupby(["aisle_id"]).\
    agg({'reordered': {'a_reorder_rt': "mean", 'a_count': "size"},\
         'add_to_cart_order': {"a_add_to_cart_order": "mean"}})

aisles_fe.columns = aisles_fe.columns.droplevel(0)
aisles_fe = aisles_fe.reset_index()

# bool_reordered = if a product is bought once, prob to be reordered at least once
aisles_fe2 = aisles_order.groupby(["user_id", "aisle_id"]).agg("size").rename("UA_nb_ordered").reset_index()
aisles_fe2["UA_bool_reordered"] = (aisles_fe2["UA_nb_ordered"] > 1).astype("int")

aisles_fe2 = aisles_fe2.groupby('aisle_id')["UA_bool_reordered"].\
    agg(["mean", "size"]).reset_index().\
    rename(index=str, columns={"mean": "a_reorder_rt_bool", "size": "a_active_user"})

aisles_fe = pd.merge(aisles_fe, aisles_fe2, how="left", on="aisle_id")
del aisles_fe2

### user_aisle_fe
print("Feature engineering on user_aisle_fe")
user_aisle_fe = aisles_order.\
    groupby(["user_id", "aisle_id"]).\
    agg({'product_id': {"UA_product_rt": "nunique"}})

user_aisle_fe.columns = user_aisle_fe.columns.droplevel(0)
user_aisle_fe = user_aisle_fe.reset_index()

user_aisle_fe = pd.merge(user_aisle_fe, users_fe[["user_id", "u_active_p"]], how="left", on="user_id")
user_aisle_fe["UA_product_rt"] = user_aisle_fe["UA_product_rt"] / user_aisle_fe["u_active_p"]
user_aisle_fe = user_aisle_fe.drop("u_active_p", axis=1)

del aisles_order

### departments_fe
print("Feature engineering on departments")

departments_order = pd.merge(order_prior, products, on="product_id")
departments_order = pd.merge(departments_order, aisles, on="aisle_id")
departments_order = pd.merge(departments_order, departments, on="department_id")

departments_fe = departments_order.\
    groupby(["department_id"]).\
    agg({'reordered': {'d_reorder_rt': "mean", 'd_count': "size"},\
         'add_to_cart_order': {"d_add_to_cart_order": "mean"}})

departments_fe.columns = departments_fe.columns.droplevel(0)
departments_fe = departments_fe.reset_index()

# bool_reordered = if a product is bought once, prob to be reordered at least once
departments_fe2 = departments_order.groupby(["user_id", "department_id"]).agg("size").rename("UD_nb_ordered").reset_index()
departments_fe2["UD_bool_reordered"] = (departments_fe2["UD_nb_ordered"] > 1).astype("int")

departments_fe2 = departments_fe2.groupby('department_id')["UD_bool_reordered"].\
    agg(["mean", "size"]).reset_index().\
    rename(index=str, columns={"mean": "d_reorder_rt_bool", "size": "d_active_user"})

departments_fe = pd.merge(departments_fe, departments_fe2, how="left", on="department_id")
del departments_fe2

### user_department_fe
print("Feature engineering on user_department_fe")
user_department_fe = departments_order.\
    groupby(["user_id", "department_id"]).\
    agg({'product_id': {"UD_product_rt": "nunique"}})

user_department_fe.columns = user_department_fe.columns.droplevel(0)
user_department_fe = user_department_fe.reset_index()

user_department_fe = pd.merge(user_department_fe, users_fe[["user_id", "u_active_p"]], how="left", on="user_id")
user_department_fe["UD_product_rt"] = user_department_fe["UD_product_rt"] / user_department_fe["u_active_p"]
user_department_fe = user_department_fe.drop("u_active_p", axis=1)

del departments_order

### Construct target Y dataset by creating user_past_product
print("Get label by creating user_past_product")
# Get for each user all the product he has already bought before
user_past_product = order_prior[["user_id", "product_id"]].drop_duplicates()

reordered_train = pd.merge(orders, order_train, on=["order_id", "user_id"])
reordered_train = reordered_train.query("reordered == 1")

user_past_product = pd.merge(user_past_product, reordered_train[["user_id", "product_id", "reordered"]],
         on=["user_id", "product_id"], how="left")

user_past_product["reordered"] = user_past_product["reordered"].fillna(0)

del reordered_train

print("Get label by creating order_none")
order_none = order_train.\
    groupby(["order_id", "user_id"])["reordered"].sum(). \
    reset_index()

order_none["reordered"] = (order_none["reordered"] < 1).astype(int)

# Create dataset which we'll learn on -------------------------------------------------
print("Create dataset which we'll learn on")
def get_df(df):
    df_set = pd.merge(df, user_past_product, on=["user_id"])
    df_set = pd.merge(df_set, users_fe, on="user_id")
    df_set = pd.merge(df_set, products_fe, how='left', on="product_id")
    df_set = pd.merge(df_set, products, how='left', on="product_id")
    df_set = pd.merge(df_set, aisles, how='left', on="aisle_id")
    df_set = pd.merge(df_set, aisles_fe, how='left', on="aisle_id")
    df_set = pd.merge(df_set, departments, how='left', on="department_id")
    df_set = pd.merge(df_set, departments_fe, how='left', on="department_id")
    df_set = pd.merge(df_set, users_products, how='left', on=["user_id", "product_id"])
    df_set = pd.merge(df_set, product2vec, how='left', on="product_id")
    df_set = pd.merge(df_set, user_aisle_fe, how='left', on=["user_id", "aisle_id"])
    df_set = pd.merge(df_set, user_department_fe, how='left', on=["user_id", "department_id"])

    # Should be done in appropriate place
    df_set["UP_rt_reordered"] = df_set["up_nb_reordered"] / df_set["order_number"] # Maybe delete because it might overfit?
    df_set["UP_rt_reordered_since_first"] = df_set["up_nb_reordered"] / df_set["up_first_order_number"]
    df_set["UP_days_no-reordered"] = df_set["up_last_order_date"] - df_set["date"]
    df_set["UP_freq_nb_no-reordered"] = df_set["up_last_order_number"] / df_set["p_freq_order"]
    df_set["UP_freq_days_no-reordered"] = df_set["UP_days_no-reordered"] / df_set["p_freq_days"]
    df_set["UP_sum_basket_rt"] = df_set["up_nb_reordered"] / df_set["U_basket_sum"]
    df_set["O_days_since_prior_order_diff"] = df_set["days_since_prior_order"] - df_set["U_days_since_mean"]
    df_set["O_days_since_prior_order_rt"] = df_set["days_since_prior_order"] / df_set["U_days_since_mean"]
    return df_set

df_train = get_df(orders.query("eval_set == 'train'"))
df_test = get_df(orders.query("eval_set == 'test'"))


def get_df_none(df):
    df_set = pd.merge(df, order_none, on=["order_id", "user_id"], how="left")
    df_set = pd.merge(df_set, users_fe, on="user_id", how="left")
    df_set = pd.merge(df_set, users_products_none, on="user_id", how="left")
    df_set["O_days_since_prior_order_diff"] = df_set["days_since_prior_order"] - df_set["U_days_since_mean"]
    df_set["O_days_since_prior_order_rt"] = df_set["days_since_prior_order"] / df_set["U_days_since_mean"]
    return df_set

df_train_none = get_df_none(orders.query("eval_set == 'train'"))
df_test_none = get_df_none(orders.query("eval_set == 'test'"))

del aisles, aisles_fe, departments, departments_fe, order_none, order_prior, order_train, orders, \
    product2vec, products, products_fe, up_fe2, users_products, users_fe, user_past_product, users_products_none
gc.collect()

# Sample by user_id ----------------------------------------
print("Sample by user_id")
user_id = df_train["user_id"].unique()
np.random.seed(seed=7)
sample_user_id = np.random.choice(user_id, size=int(len(user_id)*0.20), replace=False).tolist()

# Modeling -----------------------------------
# Sampling takes time -_-
sample_index = df_train.query("user_id == @sample_user_id").index
df_valid = df_train.ix[sample_index]
df_train = df_train.drop(sample_index)

# TODO try to not drop order_id
to_drop = ["order_id", "user_id", "eval_set", "product_id", "product_name","department", "aisle", \
           "order_number_reverse", "date", "UP_days_no-reordered"]

X_train = df_train.drop(to_drop + ["reordered"], axis=1)
X_valid = df_valid.drop(to_drop + ["reordered"], axis=1)
X_test = df_test.drop(to_drop + ["reordered"], axis=1)
y_train = df_train["reordered"]
y_valid = df_valid["reordered"]

print("Training model")
# , params= {'bin_construct_sample_cnt': 10**10}
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid)
param = {'objective': 'binary', 'metric': ['binary_logloss'], 'learning_rate':0.05, 'verbose': 0}
model_gbm = lgb.train(param, lgb_train, 100000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=150, verbose_eval=10)
#lgb.plot_importance(model_gbm, importance_type="gain")
#feature_importance = pd.DataFrame(model_gbm.feature_name())
#feature_importance.columns = ["Feature"]
#feature_importance["Importance_gain"] = model_gbm.feature_importance(importance_type='gain')
# feature_importance = feature_importance.sort("Importance_gain")
# feature_importance.head()

gc.collect()
# Modeling None -----------------------
sample_index_none = df_train_none.query("user_id == @sample_user_id").index
df_valid_none = df_train_none.ix[sample_index_none]
df_train_none = df_train_none.drop(sample_index_none)

to_drop_none = ["order_id", "user_id", "eval_set", "date"]

X_train_none = df_train_none.drop(to_drop_none + ["reordered"], axis=1)
X_valid_none = df_valid_none.drop(to_drop_none + ["reordered"], axis=1)
X_test_none = df_test_none.drop(to_drop_none + ["reordered"], axis=1)
y_train_none = df_train_none["reordered"]
y_valid_none = df_valid_none["reordered"]

print("\n\nTraining model none")
lgb_train_none = lgb.Dataset(X_train_none, label=y_train_none, max_bin=100)
lgb_valid_none = lgb.Dataset(X_valid_none, label=y_valid_none, max_bin=100)
param_none = {'objective': 'binary', 'metric': ['binary_logloss'], 'learning_rate':0.05,\
              'num_leaves':3, 'min_data_in_leaf':500, 'verbose': 0}
model_gbm_none = lgb.train(param_none, lgb_train_none, 100000, valid_sets=[lgb_train_none, lgb_valid_none], early_stopping_rounds=150, verbose_eval=10)
# lgb.plot_importance(model_gbm_none, importance_type="gain")

gc.collect()
# Predict and submit -------------------------------------------------------------
print('Predict and submit')
def get_df_pred(df, X_df, df_none, X_df_none):
    df["pred"] = model_gbm.predict(X_df, num_iteration=model_gbm.best_iteration)
    df_none["pred"] = model_gbm_none.predict(X_df_none, num_iteration=model_gbm_none.best_iteration)
    df_none["product_id"] = "None"
    df_pred_none = df_none[["order_id", "user_id", "product_id", "pred"]].copy()

    # Copy avoid warning
    df_pred = df[["order_id", "user_id", "product_id", "pred"]].copy()

    df_test_pred = pd.concat([df_pred, df_pred_none])
    return df_test_pred


def groupby_optimised_pred(group):
    group_none = group.iloc[0:1]
    none_gain = group_none["pred"].values[0]
    group_no_none = group.iloc[1:]
    group_no_none["precision"] = group_no_none["pred"].expanding().mean()
    basket_size = group_no_none["pred"].sum() #0.75 #- 0.10 # Empirically found, could be finest, f-score is asymetric
    group_no_none["recall"] = group_no_none["pred"].expanding().sum() / basket_size
    group_no_none["f_score"] = (2 * group_no_none["precision"] * group_no_none["recall"]) / (group_no_none["precision"] + group_no_none["recall"])
    f_score = group_no_none["f_score"].max()

    #group_no_none["recall_init"] = group_no_none["pred"].expanding().sum() / group_no_none["pred"].sum()
    #group_no_none["f_score_init"] = (2 * group_no_none["precision"] * group_no_none["recall_init"]) / (group_no_none["precision"] + group_no_none["recall_init"])
    #f_score_init = group_no_none["f_score_init"].max()

    max_index = np.where(group_no_none["f_score"] == f_score)[0][0]
    group_no_none = group_no_none[0:(max_index+1)] # Could be (max_index+k) with k>1 if the limit is risky maybe?

    # f_score_none is the expected f_score if we add none
    precision_none = (group_no_none["pred"].sum()) / (group_no_none.shape[0] + 1)
    recall_none = group_no_none.iloc[-1]["recall"]
    f_score_none = (2 * precision_none * recall_none) / (precision_none + recall_none)

    res = group_no_none #.drop(["precision", "recall", "f_score"], axis=1)
    # Add none if it's worth it
    # 0.07 and not 0 because f_score is under-estimated, could be finest
    if none_gain - (f_score - f_score_none) > 0.07:
        res = pd.concat([res, group_none])

    if (none_gain > f_score + 0.0):
        res = group_none

    #if (none_gain > group_no_none["pred"].values[0] + 0.11): #Worsen score, need to understand why
    #    res = group_none

    return res


def filter_optimised_pred(df):
    df["is_none"] = (df["product_id"] == 'None').astype(int)
    df = df.sort_values(["is_none", "pred"], ascending=False).drop("is_none", axis=1)
    df = df. \
        groupby("user_id"). \
        apply(groupby_optimised_pred).reset_index(drop=True)
    return df


def filter_maximize_expectation(df):
    df["is_none"] = (df["product_id"] == 'None').astype(int)
    df = df.sort_values(["is_none", "pred"], ascending=False).drop("is_none", axis=1)
    df = df. \
        groupby("user_id"). \
        apply(groupby_maximize_expectation).reset_index(drop=True)
    return df


def groupby_maximize_expectation(group):
    group_none = group.iloc[0:1]
    none_gain = group_none["pred"].values[0]
    group_no_none = group.iloc[1:]
    pred = group_no_none["pred"].values
    # Avoid weird llvm bug with numba
    if pred.shape[0] == 1:
        pred = np.append(pred, 0)

    best_k, predNone, max_f1 = utils.F1Optimizer.maximize_expectation(pred, none_gain)

    res = group_no_none.iloc[:best_k]
    if predNone:
        res = pd.concat([res, group_none])

    return res


def compute_fscore(df, df_pred):
    df_none_true = df. \
        groupby(["order_id", "user_id"])["reordered"].sum().\
        reset_index()

    # Warning bellow but don't know where :/
    # If atleast one reorderd then None is 0, otherwise 1
    df_none_true["reordered"] = (df_none_true["reordered"] < 1).astype(int)
    df_none_true.loc[:,"product_id"] = "None"
    df = pd.concat([df, df_none_true])
    df_y_true = df.query("reordered==1").groupby("user_id")["product_id"].unique().reset_index()
    df_y_true.columns = ["user_id", "y_true"]
    df_y_pred = df_pred.groupby("user_id")["product_id"].unique().reset_index()
    df_y_pred.columns = ["user_id", "y_pred"]

    res = pd.merge(df_y_true, df_y_pred, on="user_id", how="left")
    cond = res["y_pred"].isnull()
    res.loc[cond, "y_pred"] = np.array(['None'])

    res = res.apply(lambda x: utils.multilabel_fscore(x["y_true"], x["y_pred"]), axis=1)
    return res.mean()

print("Score estimation")
start = time.time()
df_valid_pred = get_df_pred(df_valid, X_valid, df_valid_none, X_valid_none)
df_valid_pred["group"] = df_valid_pred["user_id"].mod(cpu_count()*3)
print(compute_fscore(df_valid, utils.applyParallel(df_valid_pred.groupby("group"), filter_maximize_expectation)))
end = time.time()
print("Le temps d'exécution :" + str(end - start))

print("Generate submission")
df_test_pred = get_df_pred(df_test, X_test, df_test_none, X_test_none)
df_test_pred["group"] = df_test_pred["user_id"].mod(cpu_count()*3)
# utils.applyParallel(df_test_pred.groupby("group"), filter_optimised_pred).\
sub = utils.applyParallel(df_test_pred.groupby("group"), filter_maximize_expectation).\
    groupby("order_id")["product_id"].\
    apply(lambda col: col.astype(str).str.cat(sep=' ')).rename("products").reset_index()

sub = pd.merge(pd.DataFrame(df_test_pred.order_id.unique(), columns=['order_id']), sub, how="left", on="order_id")
sub = sub.fillna('None')

sub.columns = ['order_id', 'products']
sub.to_csv('./sub.csv', index=False)
gc.collect()


