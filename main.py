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
    group["date"] = group.iloc[::-1]['days_since_prior_order'].cumsum()[::-1].shift(-1).fillna(0)
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
sample_user_id = np.random.choice(user_id, size=5000, replace=False).tolist()
orders = orders.query("user_id == @sample_user_id")

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
products_fe = utils.get_products_fe(order_prior)

#products_fe_mod = utils.get_products_fe_mod(order_prior, order_train, nfold=5)

# Doesn't seem to help much?
products_organic = products[["product_id", "product_name"]].copy()
products_organic["organic_bool"] = products["product_name"].str.match("organic", case=False).astype('int')
products_fe = pd.merge(products_fe, products_organic.drop("product_name", axis=1), how="left", on="product_id")
del products_organic

### user_fe - Feature engineering on user
print("Feature engineering on user")
users_fe = utils.get_users_fe(orders, order_prior)

### user_product - UP
print("Feature engineering on User_product")
users_products = utils.get_users_products(order_prior)

### users_products_none summary of users_products
print("Feature engineering on users_products_none")
users_products_none = utils.get_users_products_none(users_products)

### aisles
print("Feature engineering on aisles")
aisles_fe = utils.get_aisles_fe(order_prior, products, aisles)

### user_aisle_fe
print("Feature engineering on user_aisle_fe")
user_aisle_fe = utils.get_user_aisle_fe(order_prior, products, aisles, users_fe)

### departments_fe
print("Feature engineering on departments")

departments_fe = utils.get_departments_fe(order_prior, products, aisles, departments)

### user_department_fe
print("Feature engineering on user_department_fe")
user_department_fe = utils.get_user_department_fe(order_prior, products, aisles, departments, users_fe)

### Construct target Y dataset by creating user_past_product
print("Get label by creating user_past_product")
# Get for each user all the product he has already bought before
user_past_product = utils.get_user_past_product(order_prior, orders, order_train)

print("Get label by creating order_none")
order_none = utils.get_order_none(order_train)

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

### Feature engineering on predicted basket
print("Feature engineering on predicted basket")
def get_mult_none(df, X_df, model_gbm):
    df_pred = df[["order_id", "user_id"]].copy()
    df_pred["pred"] = model_gbm.predict(X_df, num_iteration=model_gbm.best_iteration)
    df_pred["pred_minus"] = 1 - df_pred["pred"]

    df_pred = df_pred.groupby(["order_id", "user_id"]). \
        agg({'pred_minus': {'pred_none_prod': "prod"}, \
             'pred': {'pred_basket_sum': "sum", 'pred_basket_std':'std'}})

    df_pred.columns = df_pred.columns.droplevel(0)
    df_pred = df_pred.reset_index()

    return df_pred

def get_mult_none_cv(df_full, df_test, nfold=5):
    df_full = df_full.copy()
    df_full["fold"] = df_full["user_id"].mod(nfold)

    to_drop = ["order_id", "user_id", "eval_set", "product_id", "product_name", "department", "aisle", \
               "order_number_reverse", "date", "UP_days_no-reordered"]

    res=[]
    for fold in range(nfold):
        print("Folder: " + str(fold))
        df_valid = df_full.query("fold == @fold").drop("fold", axis=1)
        df_train = df_full.query("fold != @fold").drop("fold", axis=1)
        X_train = df_train.drop(to_drop + ["reordered"], axis=1)
        X_valid = df_valid.drop(to_drop + ["reordered"], axis=1)
        y_train = df_train["reordered"]
        y_valid = df_valid["reordered"]

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid)
        param = {'objective': 'binary', 'metric': ['binary_logloss'], 'learning_rate': 0.1, 'verbose': 0}
        model_gbm = lgb.train(param, lgb_train, 100000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=150,
                              verbose_eval=0)
        res.append(get_mult_none(df_valid, X_valid, model_gbm))

        if fold == nfold:
            X_test = df_test.drop(to_drop, axis=1)
            res.append(get_mult_none(df_test, X_test, model_gbm))

    res = pd.concat(res)
    return res

#print("get_mult_none_cv")
#mult_none_cv = get_mult_none_cv(df_train, df_test)

print("Create dataset which we'll learn on for None")
def get_df_none(df):
    df_set = pd.merge(df, order_none, on=["order_id", "user_id"], how="left")
    df_set = pd.merge(df_set, users_fe, on="user_id", how="left")
    df_set = pd.merge(df_set, users_products_none, on="user_id", how="left")
    #df_set = pd.merge(df_set, mult_none_cv, on=["order_id", "user_id"], how="left")
    df_set["O_days_since_prior_order_diff"] = df_set["days_since_prior_order"] - df_set["U_days_since_mean"]
    df_set["O_days_since_prior_order_rt"] = df_set["days_since_prior_order"] / df_set["U_days_since_mean"]
    return df_set

df_train_none = get_df_none(orders.query("eval_set == 'train'"))
df_test_none = get_df_none(orders.query("eval_set == 'test'"))

#del aisles, aisles_fe, departments, departments_fe, order_none, order_prior, order_train, orders, \
#    product2vec, products, products_fe, up_fe2, users_products, users_fe, user_past_product, users_products_none
gc.collect()

# Sample by user_id ----------------------------------------
print("Sample by user_id")
user_id = df_train["user_id"].unique()
np.random.seed(seed=7)
sample_user_id = np.random.choice(user_id, size=int(len(user_id)*0.20), replace=False).tolist()

# Modeling -----------------------------------
# Sampling takes time -_-
sample_index = df_train.query("user_id == @sample_user_id").index
df_valid = df_train.iloc[sample_index]
df_train = df_train.drop(sample_index)

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
df_valid_none = df_train_none.iloc[sample_index_none]
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
    df = df.copy()
    df_none = df_none.copy()
    df["pred"] = model_gbm.predict(X_df, num_iteration=model_gbm.best_iteration)
    df_none["pred"] = np.clip(model_gbm_none.predict(X_df_none, num_iteration=model_gbm_none.best_iteration), 0,1)
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


