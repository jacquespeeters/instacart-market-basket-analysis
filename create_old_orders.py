import pandas as pd
import numpy as np
import lightgbm as lgb
import utils
import timeit
import pickle
import gc
import time

if __name__ == '__main__':
    # Read data -----------------------------
    aisles, departments, order_prior, order_train, orders, product2vec = utils.read_data()

    print("Sample by user_id")
    user_id = orders["user_id"].unique()
    np.random.seed(seed=7)
    sample_user_id = np.random.choice(user_id, size=5000, replace=False).tolist()
    #orders = orders.query("user_id == @sample_user_id")


    def correct_orders(orders):
        orders = orders.copy()
        # Keep users we don't already have
        train_user = orders.query('eval_set == "train"')["user_id"].unique()
        orders = orders.query("user_id not in @train_user")

        # Keep users with more than 4 orders
        keep_user = orders.groupby("user_id")["order_number"].agg("size").reset_index().query("order_number > 4")["user_id"].unique()
        orders = orders.query("user_id in @keep_user")

        new_orders = orders.query("order_number_reverse !=0")
        df = orders.\
            groupby("user_id")["date"].min().rename("min_date").reset_index()

        new_orders = pd.merge(new_orders, df, on="user_id", how="left")
        new_orders["date"] = new_orders["date"] - new_orders["min_date"]
        new_orders = new_orders.drop("min_date", axis=1)

        new_orders["order_number_reverse"] = new_orders["order_number_reverse"] - 1
        cond = new_orders["order_number_reverse"] == 0
        new_orders.loc[cond,"eval_set"] = "train"
        return new_orders

    orders = correct_orders(orders)

    def correct_order_XXX(orders, order_prior):
        order_train = pd.merge(orders.query("eval_set == 'train'")[["order_id"]], order_prior, on="order_id", how="left")
        order_prior = pd.merge(orders.query("eval_set != 'train'")[["order_id"]], order_prior, on="order_id", how="left")
        return order_prior, order_train

    order_prior, order_train = correct_order_XXX(orders, order_prior)


    def generate_old_orders(orders, order_prior, order_train):
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
        products_fe = pd.merge(products_fe_mod, products_organic.drop("product_name", axis=1), how="left",
                               on="product_id")
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
        df_full = utils.get_df(orders.query("eval_set == 'train'"), user_past_product, users_fe, products_fe_mod,
                               products, aisles,
                               aisles_fe, departments, departments_fe, users_products, product2vec, user_aisle_fe,
                               user_department_fe, nfold=NFOLD)

        print("Create dataset which we'll learn on for None")
        df_full_none = utils.get_df_none(orders.query("eval_set == 'train'"), order_none, users_fe)

        return df_full, df_full_none

    df_full_old, df_full_none_old = generate_old_orders(orders, order_prior, order_train)

    print("Pickle results")
    df_full_old.to_pickle("df_full_old.p")
    df_full_none_old.to_pickle("df_full_none_old.p")
    print("End of script")
