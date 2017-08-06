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



#####################
# https://www.kaggle.com/mmueller/f1-score-expectation-maximization-in-o-n/code

import matplotlib.pylab as plt
from datetime import datetime
from numba import jit

class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    @jit
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    @jit
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)


def print_best_prediction(P, pNone=None):
    print("Maximize F1-Expectation")
    print("=" * 23)
    P = np.sort(P)[::-1]
    n = P.shape[0]
    L = ['L{}'.format(i + 1) for i in range(n)]

    if pNone is None:
        print("Estimate p(None|x) as (1-p_1)*(1-p_2)*...*(1-p_n)")
        pNone = (1.0 - P).prod()

    PL = ['p({}|x)={}'.format(l, p) for l, p in zip(L, P)]
    print("Posteriors: {} (n={})".format(PL, n))
    print("p(None|x)={}".format(pNone))

    opt = F1Optimizer.maximize_expectation(P, pNone)
    best_prediction = ['None'] if opt[1] else []
    best_prediction += (L[:opt[0]])
    f1_max = opt[2]

    print("Prediction {} yields best E[F1] of {}\n".format(best_prediction, f1_max))


def save_plot(P, filename='expected_f1.png'):
    E_F1 = pd.DataFrame(F1Optimizer.get_expectations(P).T, columns=["/w None", "/wo None"])
    best_k, _, max_f1 = F1Optimizer.maximize_expectation(P)

    plt.style.use('ggplot')
    plt.figure()
    E_F1.plot()
    plt.title('Expected F1-Score for \n {}'.format("P = [{}]".format(",".join(map(str, P)))), fontsize=12)
    plt.xlabel('k')
    plt.xticks(np.arange(0, len(P) + 1, 1.0))
    plt.ylabel('E[F1(P,k)]')
    plt.plot([best_k], [max_f1], 'o', color='#000000', markersize=4)
    plt.annotate('max E[F1(P,k)] = E[F1(P,{})] = {:.5f}'.format(best_k, max_f1), xy=(best_k, max_f1),
                 xytext=(best_k, max_f1 * 0.8), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7),
                 horizontalalignment='center', verticalalignment='top')
    plt.gcf().savefig(filename)



def timeit(P):
    s = datetime.now()
    F1Optimizer.maximize_expectation(P)
    e = datetime.now()
    return (e-s).microseconds / 1E6


def benchmark(n=100, filename='runtimes.png'):
    results = pd.DataFrame(index=np.arange(1,n+1))
    results['runtimes'] = 0

    for i in range(1,n+1):
        runtimes = []
        for j in range(5):
            runtimes.append(timeit(np.sort(np.random.rand(i))[::-1]))
        results.iloc[i-1] = np.mean(runtimes)

    x = results.index
    y = results.runtimes
    results['quadratic fit'] = np.poly1d(np.polyfit(x, y, deg=2))(x)

    plt.style.use('ggplot')
    plt.figure()
    results.plot()
    plt.title('Expectation Maximization Runtimes', fontsize=12)
    plt.xlabel('n = |P|')
    plt.ylabel('time in seconds')
    plt.gcf().savefig(filename)


#####################
from sklearn.model_selection import ParameterGrid
#from ggplot import *
import time
import lightgbm as lgb
import pandas as pd


def grid_search(lgb_train, lgb_valid, param_grid):
    """"
    param_grid = {'eta': [0.1], 'max_depth': [4, 6], 'subsample': [0.8, 1]}
    grid_search(lgb_train, lgb_valid, param_grid)
    """
    grid = ParameterGrid(param_grid)
    data = pd.DataFrame(list(grid))
    data['time'] = 0
    data['best_score'] = 0
    data['best_iter'] = 0

    for i, params in enumerate(grid):
        print("Etape " + str(i + 1))
        start = time.time()
        #ipdb.set_trace()
        model = lgb.train(params, lgb_train, 100000, valid_sets=[lgb_train, lgb_valid],
                          early_stopping_rounds=150, verbose_eval=50)

        end = time.time()
        data.loc[i,'time'] = end - start
        data.loc[i,'best_score'] = model.best_score['valid_1'][params['metric']]
        data.loc[i,'best_iter'] = model.best_iteration
    return data

#########
def get_products_fe(order_prior):
    products_fe = order_prior. \
        groupby(["product_id"]). \
        agg({'reordered': {'p_reorder_rt': "mean", 'p_count': "size"}, \
             'add_to_cart_order': {"p_add_to_cart_order": "mean"}, \
             'order_number': {"P_recency_order": "mean"}, \
             'order_number_reverse': {"P_recency_order_r": "mean"}, \
             'date': {"P_recency_date": "mean"}})

    products_fe.columns = products_fe.columns.droplevel(0)
    products_fe = products_fe.reset_index()

    # bool_reordered = if a product is bought once, prob to be reordered at least once
    up_fe2 = order_prior.groupby(["user_id", "product_id"]).agg("size").rename("up_nb_ordered").reset_index()
    up_fe2["bool_reordered"] = (up_fe2["up_nb_ordered"] > 1).astype("int")

    products_fe2 = up_fe2.groupby('product_id')["bool_reordered"]. \
        agg(["mean", "size"]).reset_index(). \
        rename(index=str, columns={"mean": "p_reorder_rt_bool", "size": "p_active_user"})

    products_fe = pd.merge(products_fe, products_fe2, how="left", on="product_id")
    del products_fe2

    # Product trend in a way
    products_trend = order_prior.query("order_number_reverse < 3"). \
        groupby(["product_id", "order_number_reverse"]).size(). \
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

    product_freq = product_freq.groupby("product_id"). \
        agg({'p_freq_days': {'p_freq_days': "mean"}, \
             'p_freq_order': {'p_freq_order': "mean"}})

    product_freq.columns = product_freq.columns.droplevel(0)
    product_freq = product_freq.reset_index()

    products_fe = pd.merge(products_fe, product_freq, how="left", on="product_id")

    del product_freq

    return products_fe


def get_products_fe_mod(order_prior, order_train, nfold=5):
    order_train = order_train.copy()
    order_train["user_fold"] = order_train["user_id"].mod(nfold)

    products_fe_mod=[]
    for fold in range(nfold):
        #print("Folder: " + str(fold))
        order_train_tmp = order_train.query("user_fold != @fold").drop("user_fold", axis=1)
        order_train_tmp = pd.concat([order_prior, order_train_tmp])
        products_fe_tmp = get_products_fe(order_train_tmp)
        products_fe_tmp["user_fold"] = fold
        products_fe_mod.append(products_fe_tmp)

    products_fe_mod = pd.concat(products_fe_mod)

    return products_fe_mod


def get_users_fe(orders, order_prior):
    users_fe = order_prior. \
        groupby("user_id"). \
        agg({'reordered': {'U_rt_reordered': 'mean'}, \
             'date': {'U_date_inscription': 'max'}})

    users_fe.columns = users_fe.columns.droplevel(0)
    users_fe = users_fe.reset_index()

    # User basket sum, mean, std
    # TODO same but only on reordered products?
    users_fe2 = order_prior. \
        groupby(["user_id", "order_id"]).size(). \
        reset_index(). \
        drop("order_id", axis=1). \
        groupby("user_id"). \
        agg([np.sum, np.mean, np.std])

    users_fe2.columns = ["U_basket_sum", "U_basket_mean", "U_basket_std"]
    users_fe2 = users_fe2.reset_index()

    users_fe = pd.merge(users_fe, users_fe2, on="user_id")
    del users_fe2

    # u_active_p == user distinct products
    # bool_reordered = if a product is bought once, prob to be reordered at least once
    up_fe2 = order_prior.groupby(["user_id", "product_id"]).agg("size").rename("up_nb_ordered").reset_index()
    up_fe2["bool_reordered"] = (up_fe2["up_nb_ordered"] > 1).astype("int")

    users_fe4 = up_fe2.groupby('user_id')["bool_reordered"]. \
        agg(["mean", "size"]).reset_index(). \
        rename(index=str, columns={"mean": "u_reorder_rt_bool", "size": "u_active_p"})

    users_fe = pd.merge(users_fe, users_fe4, on="user_id")
    del users_fe4

    users_fe5 = orders.query("order_number_reverse != 0"). \
        groupby("user_id"). \
        agg({'date': {'U_date_inscription': 'max'}, \
             'days_since_prior_order': {'U_days_since_mean': 'mean', \
                                        'U_days_since_std': 'std'}})

    users_fe5.columns = users_fe5.columns.droplevel(0)
    users_fe5 = users_fe5.reset_index()

    users_fe = pd.merge(users_fe, users_fe5, on="user_id")
    del users_fe5

    # TODO U_none_reordered_strike
    # New way
    # TODO test if it's help or not, keeping it might be used as a skrinking term
    # .query("order_number !=1")
    user_fe_none = order_prior. \
        groupby(["order_id", "user_id"]). \
        agg({'reordered': {'reordered': "sum"}, \
             'order_number_reverse': {'order_number_reverse': 'first'}})

    user_fe_none.columns = user_fe_none.columns.droplevel(0)
    user_fe_none = user_fe_none.reset_index()

    user_fe_none["reordered"] = (user_fe_none["reordered"] < 1).astype(int)
    user_fe_none["U_none_reordered_strike"] = user_fe_none["reordered"] * 1 / 2 ** (
    user_fe_none["order_number_reverse"])

    user_fe_none = user_fe_none. \
        groupby("user_id"). \
        agg({'reordered': {'U_none_reordered_mean': "mean"}, \
             'U_none_reordered_strike': {'U_none_reordered_strike': "sum"}})

    user_fe_none.columns = user_fe_none.columns.droplevel(0)
    user_fe_none = user_fe_none.reset_index()

    users_fe = pd.merge(users_fe, user_fe_none, on="user_id")
    del user_fe_none
    return users_fe


def get_users_products(order_prior):
    # Could be something else than 1/2
    order_prior["UP_date_strike"] = 1 / 2 ** (order_prior["date"] / 7)
    # order_prior["UP_order_strike"] = 100000 * 1/2 ** (order_prior["order_number_reverse"])
    order_prior["UP_order_strike"] = 1 / 2 ** (order_prior["order_number_reverse"])

    users_products = order_prior. \
        groupby(["user_id", "product_id"]). \
        agg({'reordered': {'up_nb_reordered': "size"}, \
             'add_to_cart_order': {'up_mean_add_to_cart_order': "mean"}, \
             'order_number_reverse': {'up_last_order_number': "min", 'up_first_order_number': "max"}, \
             'date': {'up_last_order_date': "min", 'up_first_date_number': "max"}, \
             'UP_date_strike': {"UP_date_strike": "sum"}, \
             'UP_order_strike': {"UP_order_strike": "sum"}})

    users_products.columns = users_products.columns.droplevel(0)
    users_products = users_products.reset_index()
    return users_products


def get_users_products_none(users_products):
    users_products_none = users_products.groupby("user_id"). \
        agg({'UP_date_strike': {'O_date_strike_max': "max", 'O_date_strike_sum': "sum", 'O_date_strike_mean': "mean"}, \
             'UP_order_strike': {'O_order_strike_max': "max", 'O_order_strike_sum': "sum",
                                 'O_date_order_mean': "mean"}})

    users_products_none.columns = users_products_none.columns.droplevel(0)
    users_products_none = users_products_none.reset_index()
    return users_products_none


def get_aisles_fe(order_prior, products, aisles):
    aisles_order = pd.merge(order_prior, products, on="product_id")
    aisles_order = pd.merge(aisles_order, aisles, on="aisle_id")

    aisles_fe = aisles_order. \
        groupby(["aisle_id"]). \
        agg({'reordered': {'a_reorder_rt': "mean", 'a_count': "size"}, \
             'add_to_cart_order': {"a_add_to_cart_order": "mean"}})

    aisles_fe.columns = aisles_fe.columns.droplevel(0)
    aisles_fe = aisles_fe.reset_index()

    # bool_reordered = if a product is bought once, prob to be reordered at least once
    aisles_fe2 = aisles_order.groupby(["user_id", "aisle_id"]).agg("size").rename("UA_nb_ordered").reset_index()
    aisles_fe2["UA_bool_reordered"] = (aisles_fe2["UA_nb_ordered"] > 1).astype("int")

    aisles_fe2 = aisles_fe2.groupby('aisle_id')["UA_bool_reordered"]. \
        agg(["mean", "size"]).reset_index(). \
        rename(index=str, columns={"mean": "a_reorder_rt_bool", "size": "a_active_user"})

    aisles_fe = pd.merge(aisles_fe, aisles_fe2, how="left", on="aisle_id")
    del aisles_fe2
    return aisles_fe


def get_user_aisle_fe(order_prior, products, aisles, users_fe):
    aisles_order = pd.merge(order_prior, products, on="product_id")
    aisles_order = pd.merge(aisles_order, aisles, on="aisle_id")
    user_aisle_fe = aisles_order. \
        groupby(["user_id", "aisle_id"]). \
        agg({'product_id': {"UA_product_rt": "nunique"}})

    user_aisle_fe.columns = user_aisle_fe.columns.droplevel(0)
    user_aisle_fe = user_aisle_fe.reset_index()

    user_aisle_fe = pd.merge(user_aisle_fe, users_fe[["user_id", "u_active_p"]], how="left", on="user_id")
    user_aisle_fe["UA_product_rt"] = user_aisle_fe["UA_product_rt"] / user_aisle_fe["u_active_p"]
    user_aisle_fe = user_aisle_fe.drop("u_active_p", axis=1)
    return user_aisle_fe


def get_departments_fe(order_prior, products, aisles, departments):
    departments_order = pd.merge(order_prior, products, on="product_id")
    departments_order = pd.merge(departments_order, aisles, on="aisle_id")
    departments_order = pd.merge(departments_order, departments, on="department_id")

    departments_fe = departments_order. \
        groupby(["department_id"]). \
        agg({'reordered': {'d_reorder_rt': "mean", 'd_count': "size"}, \
             'add_to_cart_order': {"d_add_to_cart_order": "mean"}})

    departments_fe.columns = departments_fe.columns.droplevel(0)
    departments_fe = departments_fe.reset_index()

    # bool_reordered = if a product is bought once, prob to be reordered at least once
    departments_fe2 = departments_order.groupby(["user_id", "department_id"]).agg("size").rename(
        "UD_nb_ordered").reset_index()
    departments_fe2["UD_bool_reordered"] = (departments_fe2["UD_nb_ordered"] > 1).astype("int")

    departments_fe2 = departments_fe2.groupby('department_id')["UD_bool_reordered"]. \
        agg(["mean", "size"]).reset_index(). \
        rename(index=str, columns={"mean": "d_reorder_rt_bool", "size": "d_active_user"})

    departments_fe = pd.merge(departments_fe, departments_fe2, how="left", on="department_id")
    del departments_fe2
    return departments_fe


def get_user_department_fe(order_prior, products, aisles, departments, users_fe):
    departments_order = pd.merge(order_prior, products, on="product_id")
    departments_order = pd.merge(departments_order, aisles, on="aisle_id")
    departments_order = pd.merge(departments_order, departments, on="department_id")

    user_department_fe = departments_order. \
        groupby(["user_id", "department_id"]). \
        agg({'product_id': {"UD_product_rt": "nunique"}})

    user_department_fe.columns = user_department_fe.columns.droplevel(0)
    user_department_fe = user_department_fe.reset_index()

    user_department_fe = pd.merge(user_department_fe, users_fe[["user_id", "u_active_p"]], how="left", on="user_id")
    user_department_fe["UD_product_rt"] = user_department_fe["UD_product_rt"] / user_department_fe["u_active_p"]
    user_department_fe = user_department_fe.drop("u_active_p", axis=1)
    return user_department_fe


def get_user_past_product(order_prior, orders, order_train):
    user_past_product = order_prior[["user_id", "product_id"]].drop_duplicates()

    reordered_train = pd.merge(orders, order_train, on=["order_id", "user_id"])
    reordered_train = reordered_train.query("reordered == 1")

    user_past_product = pd.merge(user_past_product, reordered_train[["user_id", "product_id", "reordered"]],
                                 on=["user_id", "product_id"], how="left")

    user_past_product["reordered"] = user_past_product["reordered"].fillna(0)
    return user_past_product

def get_order_none(order_train):
    order_none = order_train. \
        groupby(["order_id", "user_id"])["reordered"].sum(). \
        reset_index()

    order_none["reordered"] = (order_none["reordered"] < 1).astype(int)
    return order_none
