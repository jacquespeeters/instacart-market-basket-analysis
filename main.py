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

if __name__ == '__main__':
    # Read data -----------------------------
    aisles, departments, order_prior, order_train, orders, products, product2vec = utils.read_data()

    # Data wrangling ----------------------------------------------------

    print("Sample by user_id")
    user_id = orders["user_id"].unique()
    np.random.seed(seed=7)
    sample_user_id = np.random.choice(user_id, size=10000, replace=False).tolist()
    #orders = orders.query("user_id == @sample_user_id")

    ### Feature engineering start here
    print("Add user_id to order_products__XXX ")
    order_prior = pd.merge(orders, order_prior, on=["order_id"])
    order_train = pd.merge(orders, order_train, on=["order_id"])

    print("Add order_size to order_prior")
    order_stat = order_prior.groupby('order_id').agg({'order_id': 'size'}) \
        .rename(columns={'order_id': 'order_size'}).reset_index()
    order_prior = pd.merge(order_prior, order_stat, on='order_id')
    order_prior['add_to_cart_order_inverted'] = order_prior.order_size - order_prior.add_to_cart_order
    order_prior['add_to_cart_order_relative'] = order_prior.add_to_cart_order / order_prior.order_size
    del order_stat

    print("Add last_basket_size to orders ")
    last_basket_size = order_prior.groupby(["user_id", "order_number_reverse"]).size().\
        rename("last_basket_size").reset_index()
    last_basket_size["order_number_reverse"] = last_basket_size["order_number_reverse"] - 1

    orders = pd.merge(orders, last_basket_size, how="left", on=["user_id", "order_number_reverse"])

    del last_basket_size

    ### products_fe - Feature engineering on products
    print("Feature engineering on products")
    #products_fe = utils.get_products_fe(order_prior)

    NFOLD = 5
    products_fe_mod = utils.get_products_fe_mod(order_prior, order_train, nfold=NFOLD)

    # Doesn't seem to help much?
    products_organic = products[["product_id", "product_name"]].copy()
    products_organic["organic_bool"] = products["product_name"].str.match("organic", case=False).astype('int')
    products_fe = pd.merge(products_fe_mod, products_organic.drop("product_name", axis=1), how="left", on="product_id")
    del products_organic

    ### user_fe - Feature engineering on user
    print("Feature engineering on user")
    users_fe = utils.get_users_fe(orders, order_prior)

    ### user_product - UP
    print("Feature engineering on User_product")
    users_products = utils.get_users_products(order_prior)

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
    df_full = utils.get_df(orders.query("eval_set == 'train'"), user_past_product, users_fe, products_fe_mod, products, aisles,
               aisles_fe, departments, departments_fe, users_products, product2vec, user_aisle_fe,
               user_department_fe, nfold=NFOLD)

    df_test = utils.get_df(orders.query("eval_set == 'test'"), user_past_product, users_fe, products_fe_mod, products, aisles,
               aisles_fe, departments, departments_fe, users_products, product2vec, user_aisle_fe,
               user_department_fe, nfold=NFOLD)


    print("Create dataset which we'll learn on for None")
    df_full_none = utils.get_df_none(orders.query("eval_set == 'train'"), order_none, users_fe)
    df_test_none = utils.get_df_none(orders.query("eval_set == 'test'"), order_none, users_fe)

    print("Add old dataset")
    df_full_old = pd.read_pickle("df_full_old.p")
    df_full_none_old = pd.read_pickle("df_full_none_old.p")
    df_full, df_full_none = utils.add_old_orders(df_full, df_full_none, df_full_old, df_full_none_old)
    del df_full_old, df_full_none_old
    gc.collect()

    ### Feature engineering on predicted basket
    print("Feature engineering on predicted basket")
    param = {'objective': 'binary', 'metric': ['binary_logloss'], 'learning_rate': 0.025, 'verbose': 0}
    mult_none_cv = utils.get_mult_none_cv(df_full, df_test, param, early_stopping_rounds=50, nfold=5)

    df_full_none = utils.get_df_none_add(df_full_none, mult_none_cv)
    df_test_none = utils.get_df_none_add(df_test_none, mult_none_cv)

    #del aisles, aisles_fe, departments, departments_fe, order_none, order_prior, order_train, orders, \
    #    product2vec, products, products_fe, up_fe2, users_products, users_fe, user_past_product, users_products_none
    gc.collect()

    # Sample by user_id ----------------------------------------
    valid_fold = 10

    # Modeling -----------------------------------
    # TODO split on order_id
    df_valid = df_full.query("user_id % @valid_fold == 0")
    df_train = df_full.query("user_id % @valid_fold != 0")

    to_drop = ["order_id", "user_id", "eval_set", "product_id", "product_name","department", "aisle", \
               "order_number_reverse", "date", "UP_days_no-reordered"]

    X_train = df_train.drop(to_drop + ["reordered"], axis=1)
    X_valid = df_valid.drop(to_drop + ["reordered"], axis=1)
    X_full = df_full.drop(to_drop + ["reordered"], axis=1)
    X_test = df_test.drop(to_drop + ["reordered"], axis=1)
    y_train = df_train["reordered"]
    y_valid = df_valid["reordered"]
    y_full = df_full["reordered"]

    print("Training model")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)

    param = {'objective': 'binary', 'metric': ['binary_logloss'], 'learning_rate':0.025, 'verbose': 0}
    model_gbm = lgb.train(param, lgb_train, 100000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=150, verbose_eval=100)
    #lgb.plot_importance(model_gbm, importance_type="gain")
    # feature_importance = pd.DataFrame({'feature':model_gbm.feature_name(), 'importances': model_gbm.feature_importance(importance_type='gain')})
    # feature_importance = feature_importance.sort_values('importances', ascending=False)
    # feature_importance["rank"] = feature_importance.expanding()["feature"].count()
    # feature_importance.head()

    gc.collect()

    # Modeling None -----------------------
    df_valid_none = df_full_none.query("user_id % @valid_fold == 0")
    df_train_none = df_full_none.query("user_id % @valid_fold != 0")

    to_drop_none = ["order_id", "user_id", "eval_set", "date"]

    X_train_none = df_train_none.drop(to_drop_none + ["reordered"], axis=1)
    X_valid_none = df_valid_none.drop(to_drop_none + ["reordered"], axis=1)
    X_test_none = df_test_none.drop(to_drop_none + ["reordered"], axis=1)
    X_full_none = df_full_none.drop(to_drop_none + ["reordered"], axis=1)
    y_train_none = df_train_none["reordered"]
    y_valid_none = df_valid_none["reordered"]
    y_full_none = df_full_none["reordered"]

    print("\n\nTraining model none")
    lgb_train_none = lgb.Dataset(X_train_none, label=y_train_none, max_bin=100)
    lgb_valid_none = lgb.Dataset(X_valid_none, label=y_valid_none, max_bin=100)
    param_none = {'objective': 'binary', 'metric': ['binary_logloss'], 'learning_rate':0.05,\
                  'num_leaves':3, 'min_data_in_leaf':500, 'verbose': 0}
    model_gbm_none = lgb.train(param_none, lgb_train_none, 100000, valid_sets=[lgb_train_none, lgb_valid_none], early_stopping_rounds=150, verbose_eval=100)
    # lgb.plot_importance(model_gbm_none, importance_type="gain")
    #feature_importance = pd.DataFrame({'feature':model_gbm_none.feature_name(), 'importances': model_gbm_none.feature_importance(importance_type='gain')})
    #feature_importance.sort_values('importances')

    gc.collect()

    print("Score estimation")
    start = time.time()
    df_valid_pred = utils.get_df_pred(df_valid, X_valid, df_valid_none, X_valid_none, model_gbm, model_gbm_none)
    df_valid_pred["group"] = df_valid_pred["user_id"].mod(cpu_count()*3)
    print(utils.compute_fscore(df_valid, utils.applyParallel(df_valid_pred.groupby("group"), utils.filter_maximize_expectation)))
    end = time.time()
    print("Le temps d'exécution :" + str(end - start))

    print("Generate submission")
    df_test_pred = utils.get_df_pred(df_test, X_test, df_test_none, X_test_none, model_gbm, model_gbm_none)
    df_test_pred["group"] = df_test_pred["user_id"].mod(cpu_count()*3)
    # utils.applyParallel(df_test_pred.groupby("group"), utils.filter_optimised_pred).\
    sub = utils.applyParallel(df_test_pred.groupby("group"), utils.filter_maximize_expectation).\
        groupby("order_id")["product_id"].\
        apply(lambda col: col.astype(str).str.cat(sep=' ')).rename("products").reset_index()

    sub = pd.merge(pd.DataFrame(df_test_pred.order_id.unique(), columns=['order_id']), sub, how="left", on="order_id")
    sub = sub.fillna('None')

    sub.columns = ['order_id', 'products']
    sub.to_csv('./sub.csv', index=False)
    gc.collect()


