import pandas as pd
import numpy as np
import lightgbm as lgb
import utils
import ipdb
import timeit
import pickle
import sklearn as sk
import gc

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
last_basket_size = order_prior.groupby(["user_id", "order_number_reverse"]).size().\
    rename("last_basket_size").reset_index()
last_basket_size["order_number_reverse"] = last_basket_size["order_number_reverse"] - 1

orders = pd.merge(orders, last_basket_size, how="left", on=["user_id", "order_number_reverse"])

del last_basket_size

### products_fe - Feature engineering on products
print("Feature engineering on products")

products_fe = order_prior.\
    groupby(["product_id"]).\
    agg({'reordered': {'p_reorder_rt': "mean", 'p_count': "size"},\
         'add_to_cart_order': {"p_add_to_cart_order": "mean"},\
         'order_number': {"P_recency_order": "mean"}, \
         'order_number_reverse': {"P_recency_order_r": "mean"}, \
         'date': {"P_recency_date": "mean"}})

#products_fe = order_prior.\
#    groupby(["product_id"]).\
#    agg({'reordered': {'p_reorder_rt': "mean", 'p_count': "size"},\
#         'add_to_cart_order': {"p_add_to_cart_order": "mean"}})

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

# days_since_prior_order info
users_fe = orders.\
    groupby("user_id")["days_since_prior_order"].\
    agg([np.sum, np.mean, np.std])

users_fe.columns = ["sum_days_since_prior_order", "mean_days_since_prior_order", "std_days_since_prior_order"]
users_fe = users_fe.reset_index()

# User basket sum, mean, std
# TODO same but only on reordered products
users_fe2 = order_prior.\
    groupby(["user_id", "order_id"]).size().\
    reset_index().\
    drop("order_id", axis=1).\
    groupby("user_id").\
    agg([np.sum, np.mean, np.std])

users_fe2.columns = ["sum_basket", "mean_basket", "std_basket"]
users_fe2 = users_fe2.reset_index()

users_fe3 = order_prior.\
    groupby("user_id")["reordered"].\
    agg("mean").\
    reset_index()

users_fe3.columns = ["user_id", "U_rt_reordered"]

users_fe = pd.merge(users_fe, users_fe2, on="user_id")
users_fe = pd.merge(users_fe, users_fe3, on="user_id")
del users_fe2, users_fe3

# u_active_p == user distinct products
users_fe4 = up_fe2.groupby('user_id')["bool_reordered"].\
    agg(["mean","size"]).reset_index().\
    rename(index=str, columns={"mean": "u_reorder_rt_bool", "size": "u_active_p"})

users_fe = pd.merge(users_fe, users_fe4, on="user_id")
del users_fe4

user_fe_none = order_prior.\
    groupby(["order_id", "user_id"])["reordered"].sum(). \
    reset_index()

user_fe_none["reordered"] = (user_fe_none["reordered"] < 1).astype(int)

user_fe_none = user_fe_none.\
    groupby("user_id")["reordered"]. \
    agg({'reordered': {'U_none_reordered_mean': "mean", 'U_none_reordered_sum': "sum"}})

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
users_products["UP_order_strike_rt"] = users_products["UP_order_strike"] / ((1 - 1/2**(users_products["up_first_order_number"] + 1))/(1-1/2) - 1)

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
departments_order.head()
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
    df_set["up_rt_reordered"] = df_set["up_nb_reordered"] / df_set["order_number"] # Maybe delete because it might overfit?
    df_set["up_rt_reordered_since_first"] = df_set["up_nb_reordered"] / df_set["up_first_order_number"]
    df_set["up_days_no-reordered"] = df_set["up_last_order_date"] - df_set["date"]
    df_set["up_freq_nb_no-reordered"] = df_set["up_last_order_number"] / df_set["p_freq_order"]
    df_set["up_freq_days_no-reordered"] = df_set["up_days_no-reordered"] / df_set["p_freq_days"]
    return df_set

df_train = get_df(orders.query("eval_set == 'train'"))
df_test = get_df(orders.query("eval_set == 'test'"))


def get_df_none(df):
    df_set = pd.merge(df, order_none, on=["order_id", "user_id"], how="left")
    df_set = pd.merge(df_set, users_fe, on="user_id", how="left")
    df_set = pd.merge(df_set, users_products_none, on="user_id", how="left")
    return df_set

df_train_none = get_df_none(orders.query("eval_set == 'train'"))
df_test_none = get_df_none(orders.query("eval_set == 'test'"))

#del aisles, aisles_fe, departments, departments_fe, order_none, order_prior, order_train, orders, \
#    product2vec, products, products_fe, up_fe2, users_products, users_fe, user_past_product, users_products_none
gc.collect()

# Modeling -----------------------
print("Sample by user_id")
user_id = df_train["user_id"].unique()
np.random.seed(seed=7)
sample_user_id = np.random.choice(user_id, size=int(len(user_id)*0.20), replace=False).tolist()
# Sampling takes time -_-
sample_index = df_train.query("user_id == @sample_user_id").index
df_valid = df_train.ix[sample_index]

to_drop = ["order_id", "user_id", "eval_set", "product_id", "product_name","department", "aisle", \
           "order_number_reverse", "date"]

X_train = df_train.drop(to_drop + ["reordered"], axis=1)
X_test = df_test.drop(to_drop + ["reordered"], axis=1)
y_train = df_train["reordered"]

X_valid = X_train.ix[sample_index]
y_valid = y_train.ix[sample_index]
X_train = X_train.drop(sample_index)
y_train = y_train.drop(sample_index)

print("Training model")
# , params= {'bin_construct_sample_cnt': 10**10}
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid)
param = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate':0.05}
model_gbm = lgb.train(param, lgb_train, 100000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=100, verbose_eval=10)

#lgb.plot_importance(model_gbm, importance_type="gain")
#model_gbm.feature_importance(importance_type='gain')
#feature_importance = pd.DataFrame(model_gbm.feature_name())
#feature_importance.columns = ["Feature"]
#feature_importance["Importance_gain"] = model_gbm.feature_importance(importance_type='gain')
#feature_importance.head()

# Modeling None -----------------------

sample_index_none = df_train_none.query("user_id == @sample_user_id").index
df_valid_none = df_train_none.ix[sample_index_none]

to_drop_none = ["order_id", "user_id", "eval_set", "date"]

X_train_none = df_train_none.drop(to_drop_none + ["reordered"], axis=1)
X_test_none = df_test_none.drop(to_drop_none + ["reordered"], axis=1)
y_train_none = df_train_none["reordered"]

X_valid_none = X_train_none.ix[sample_index_none]
y_valid_none = y_train_none.ix[sample_index_none]
X_train_none = X_train_none.drop(sample_index_none)
y_train_none = y_train_none.drop(sample_index_none)

print("\n\nTraining model")
lgb_train_none = lgb.Dataset(X_train_none, label=y_train_none)
lgb_valid_none = lgb.Dataset(X_valid_none, label=y_valid_none)
param_none = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate':0.05}
model_gbm_none = lgb.train(param_none, lgb_train_none, 100000, valid_sets=[lgb_train_none, lgb_valid_none], early_stopping_rounds=50, verbose_eval=10)
#lgb.plot_importance(model_gbm_none, importance_type="gain")


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
    group_none = group.query("product_id == 'None'")
    none_gain = group_none["pred"].values[0]

    group_no_none = group.query("product_id != 'None'")
    group_no_none = group_no_none.sort_values(["pred"], ascending=False)
    group_no_none["precision"] = group_no_none["pred"].expanding().mean()
    basket_size = group_no_none["pred"].sum()
    group_no_none["recall"] = group_no_none["pred"].expanding().sum() / basket_size
    group_no_none["f_score"] = (2 * group_no_none["precision"] * group_no_none["recall"]) / (group_no_none["precision"] + group_no_none["recall"])
    f_score = group_no_none["f_score"].max()
    max_index = np.where(group_no_none["f_score"] == f_score)[0][0]
    group_no_none = group_no_none[0:(max_index+1)] # Could be (max_index+k) with k>1 if the limit is risky maybe?

    # f_score_none is the expected f_score if we add none
    precision_none = (group_no_none["pred"].sum()) / (group_no_none.shape[0] + 1)
    recall_none = group_no_none.iloc[-1]["recall"]
    f_score_none = (2 * precision_none * recall_none) / (precision_none + recall_none)

    #res = pd.concat([group_no_none, group_none.query("pred > @TRESHOLD")]).drop(["precision", "recall", "f_score"], axis=1)
    res = group_no_none.drop(["precision", "recall", "f_score"], axis=1)

    # TODO add none if it's worth it
    #if (none_gain - (f_score - f_score_none) > seuil) and (none_gain < max(group_no_none["pred"])):
    #    print(none_gain - (f_score - f_score_none))

    if (none_gain - (f_score - f_score_none) > 0.07):
        res = pd.concat([res, group_none])

    #    print(none_gain - (f_score-f_score_none))
    #    res = pd.concat([res,group_none])

    # TODO improve condition
    if none_gain > f_score:
    #if none_gain > (none_gain + f_score_none - 0.07):
        res = group_none
    #else :
    #    res = group_no_none[0:(max_index+1)].drop(["precision", "recall", "f_score"], axis=1)

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

    # Warning bellow but don't know where :/
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

    res = pd.merge(df_y_true, df_y_pred, on="user_id", how="left")
    cond = res["y_pred"].isnull()
    res.loc[cond, "y_pred"] = np.array(['None'])

    res = res.apply(lambda x: utils.multilabel_fscore(x["y_true"], x["y_pred"]), axis=1)
    return res.mean()

print("Score estimation")
df_valid_pred = get_df_pred(df_valid, X_valid, df_valid_none, X_valid_none)
print(compute_fscore(filter_optimised_pred(df_valid_pred)))

df_test_pred = get_df_pred(df_test, X_test, df_test_none, X_test_none)

print("Generate submission")
sub = filter_optimised_pred(df_test_pred).\
    groupby("order_id")["product_id"].\
    apply(lambda col: col.astype(str).str.cat(sep=' ')).rename("products").reset_index()

sub = pd.merge(pd.DataFrame(df_test_pred.order_id.unique(), columns=['order_id']), sub, how="left", on="order_id")
sub = sub.fillna('None')

sub.columns = ['order_id', 'products']
sub.to_csv('./sub.csv', index=False)

