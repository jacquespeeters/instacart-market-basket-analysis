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
