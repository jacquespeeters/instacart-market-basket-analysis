import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Thanks for the inspiration
# https://www.kaggle.com/paulantoine/light-gbm-benchmark-0-3692/code

# Read data -----------------------------
aisles = pd.read_csv("./data/aisles.csv")
aisles.head()

departments = pd.read_csv("./data/departments.csv")
departments.head()

order_products__prior = pd.read_csv("./data/order_products__prior.csv")
order_products__prior.head()

order_products__train = pd.read_csv("./data/order_products__train.csv")
order_products__train.head()

orders = pd.read_csv("./data/orders.csv")
orders.head()

products = pd.read_csv("./data/products.csv")
products.head()

sample_submission = pd.read_csv("./data/sample_submission.csv")
sample_submission.head()


# Data wrangling ----------------------------------------------------

print("Sample by user_id")
user_id = orders["user_id"].unique()
sample_user_id = np.random.choice(user_id, size=1000, replace=False).tolist()
#orders = orders.query("user_id == @sample_user_id")

###

print("Add user_id to order_products__XXX ")
order_products__prior = pd.merge(orders[["order_id", "user_id"]], order_products__prior, on="order_id")
order_products__train = pd.merge(orders[["order_id", "user_id"]], order_products__train, on="order_id")


### feature engineering on products
print("feature engineering on products")
products_fe = pd.DataFrame()

products_fe["orders"] = order_products__prior.\
    groupby("product_id").\
    size()

products_fe["reorders"] = order_products__prior.\
    query("reordered == 1").\
    groupby("product_id").\
    size()

# TODO : haha l'indicateur est pas bon du tout ici :D
products_fe['reorder_rate'] = products_fe["reorders"] / products_fe["orders"]

order_products__prior.head()

products_fe["add_to_cart_order"] = order_products__prior.\
    groupby("product_id")["add_to_cart_order"].\
    mean()

products_fe = products_fe.reset_index()

### Feature engineering on user
print("Feature engineering on user")

# days_since_prior_order info
users_fe = orders.\
    groupby("user_id")["days_since_prior_order"].\
    agg([np.sum, np.mean, np.std])

users_fe.columns = ["sum_days_since_prior_order", "mean_days_since_prior_order", "std_days_since_prior_order"]
users_fe = users_fe.reset_index()
users_fe.head()
orders.head()

# Basket info
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

### user_past_product
print("user_past_product")
#order_products__prior["order_number_reverse"] = order_products__prior.\
#    groupby(by="user_id", group_keys=False).\
#    apply(lambda g: g["order_number"].max() - g["order_number"])

# Get for each user all the product he has already bought before
#data.query("order_number_reverse > -1"). # Can be change if we want new samples
user_past_product = order_products__prior.\
    groupby("user_id")["product_id"].unique().\
    reset_index()

# Reshaping => equivalent to explode function
rows = []
for i, row in user_past_product.iterrows():
    for product in row.product_id:
        rows.append([row.user_id , product])

user_past_product = pd.DataFrame(rows, columns=user_past_product.columns)
user_past_product.head()

#TODO : Add None (i.e 99999) product if an order has no reordered in order_products__XXX
# rajouter le produit None si commande ne possède aucun reordered dans order_products__XXX

reordered_train = pd.merge(orders, order_products__train, on=["order_id", "user_id"])
reordered_train = reordered_train.query("reordered == 1")
reordered_train.head()

user_past_product = pd.merge(user_past_product, reordered_train[["user_id", "product_id", "reordered"]],
         on=["user_id", "product_id"], how="left")

user_past_product["reordered"] = user_past_product["reordered"].fillna(0)

### user_product
print("User_product")

users_products = order_products__prior.\
    groupby(["user_id", "product_id"]).\
    agg({'reordered' : {'nb_orders': "size"} ,\
         'add_to_cart_order' : {'mean_add_to_cart_order': "mean"}})

users_products.columns = users_products.columns.droplevel(0)
users_products = users_products.reset_index()
users_products.head()


# Create dataset which we'll learn on -------------------------------------------------

def get_df(df):
    df_set = pd.merge(df, \
                      user_past_product, on=["user_id"])
    df_set = pd.merge(df_set, users_fe, on="user_id")
    df_set = pd.merge(df_set, products_fe, on="product_id")
    df_set = pd.merge(df_set, products, on="product_id")
    df_set = pd.merge(df_set, aisles, on="aisle_id")
    df_set = pd.merge(df_set, departments, on="department_id")
    df_set = pd.merge(df_set, users_products, on=["user_id", "product_id"])

    df_set["rt_orders"] =  df_set["nb_orders"] / df_set["order_number"]

    return df_set

df_train = get_df(orders.query("eval_set == 'train'"))
df_test = get_df(orders.query("eval_set == 'test'"))

# Modeling -----------------------
to_drop = ["order_id", "user_id", "eval_set", "product_id", "product_name","department", "aisle"]

X_train = df_train.drop(to_drop + ["reordered"], axis=1)
X_test = df_test.drop(to_drop + ["reordered"], axis=1)
y_train = df_train["reordered"]

# TODO replace by default lightgbm app and not the sklearn API
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=200000)

# TODO: sample by user
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

model_gbm = gbm.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric='l1',
        early_stopping_rounds=5)

# Predict and submit -------------------------------------------------------------
print('Predict and submit')
df_test["reordered"] = model_gbm.predict(X_test)
df_test.head()


def precision_mean(x):
    x["precision_mean"] = pd.expanding_mean(x["reordered"])
    return x

def recall_mean(x):
    x["recall"] = pd.expanding_sum(x["reordered"]) / sum(x["reordered"])
    return x

# Test if possible to optimize F1 by estimating recall and precision given prediction

#df_test[["user_id", "reordered"]].\
#    sort(["reordered"]).\
#    groupby("user_id").\
#    apply(recall_mean)

#df_test[["user_id", "reordered"]].\
#    sort(["reordered"]).\
#    groupby("user_id").apply(lambda x: pd.expanding_mean(x["reordered"]))

TRESHOLD = 0.22 # guess, should be tuned with crossval on a subset of train data
d = dict()
for row in df_test.itertuples():
    if row.reordered > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in df_test.order_id:
    if order not in d:
        d[order] = 'None'

# All users in df_test had a previous order

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('./sub.csv', index=False)
