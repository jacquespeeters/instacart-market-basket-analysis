import pandas as pd
import numpy as np
import lightgbm as lgb
import utils
import timeit
import pickle
import gc
import time

# Read data -----------------------------
# TODO function to read csv
aisles = pd.read_csv("./data/aisles.csv")
departments = pd.read_csv("./data/departments.csv")
order_prior = pd.read_csv("./data/order_products__prior.csv")
order_train = pd.read_csv("./data/order_products__train.csv")
orders = pd.read_csv("./data/orders.csv")
products = pd.read_csv("./data/products.csv")

#orders = orders.groupby("user_id").\
#    apply(add_fe_to_orders)

#pickle.dump(orders, open("orders.p", "wb"))
orders = pickle.load(open("orders.p", "rb"))
product2vec = pickle.load(open("product2vec.p", "rb"))

def correct_orders(orders):
    new_orders = orders.query("order_number_reverse !=0").copy()
    df = orders.\
        groupby("user_id")["date"].min().rename("min_date").reset_index()

    new_orders = pd.merge(new_orders, df, on="user_id", how="left")
    new_orders["date"] = new_orders["date"] - new_orders["min_date"]
    new_orders = new_orders.drop("min_date", axis=1)

    new_orders["order_number_reverse"] = new_orders["order_number_reverse"] - 1
    cond = new_orders["order_number_reverse"] == 0
    new_orders.loc[cond,"eval_set"] = "train"
    # TODO delete user with less than 4 orders?
    return new_orders

orders = correct_orders(orders)

orders.query("eval_set == ['train', 'test']").order_number.describe()
correct_orders(orders).query("eval_set == 'train'").order_number.describe()

def correct_order_XXX(orders, order_prior):
    order_train = pd.merge(orders.query("eval_set == 'train'")[["order_id"]], order_prior, on="order_id", how="left")
    order_prior = pd.merge(orders.query("eval_set != 'train'")[["order_id"]], order_prior, on="order_id", how="left")
    return order_prior, order_train

order_prior, order_train = correct_order_XXX(orders, order_prior)


def generate_old_orders(orders, order_prior, order_train):
    print("Add user_id to order_products__XXX ")
    order_prior = pd.merge(orders, order_prior, on=["order_id"])
    order_train = pd.merge(orders, order_train, on=["order_id"])

    # Add last_basket_size
    last_basket_size = order_prior.groupby(["user_id", "order_number_reverse"]).size(). \
        rename("last_basket_size").reset_index()
    last_basket_size["order_number_reverse"] = last_basket_size["order_number_reverse"] - 1

    orders = pd.merge(orders, last_basket_size, how="left", on=["user_id", "order_number_reverse"])

    del last_basket_size

    ### products_fe - Feature engineering on products
    print("Feature engineering on products")
    # products_fe = utils.get_products_fe(order_prior)

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
    df_full = utils.get_df(orders.query("eval_set == 'train'"), user_past_product, users_fe, products_fe_mod, products,
                           aisles,
                           aisles_fe, departments, departments_fe, users_products, product2vec, user_aisle_fe,
                           user_department_fe, nfold=NFOLD)

    print("Create dataset which we'll learn on for None")
    df_full_none = utils.get_df_none(orders.query("eval_set == 'train'"), order_none, users_fe)

    return df_full, df_full_none

df_full_old, df_full_none_old = generate_old_orders(orders, order_prior, order_train)


df_full_old.to_pickle("df_full_old.p")
df_full_none_old.to_pickle("df_full_none_old.p")

