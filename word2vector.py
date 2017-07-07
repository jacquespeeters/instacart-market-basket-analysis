import pandas as pd
import numpy as np
import gensim
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


train_orders = pd.read_csv("./data/order_products__train.csv")
prior_orders = pd.read_csv("./data/order_products__prior.csv")
products = pd.read_csv("./data/products.csv").set_index('product_id')

# Turn the product ID to a string
train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

# Extract the ordered products in each order

train_products = train_orders.groupby("order_id")["product_id"].unique()
prior_products = prior_orders.groupby("order_id")["product_id"].unique()
#train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
#prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

# Create the final sentences
sentences = prior_products.append(train_products).values

# Train Word2Vec model
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

# Organize data for visualization
vocab = list(model.wv.vocab.keys())

model.wv[vocab]
suggestions = model.most_similar(positive=[vocab[1]], topn=5)

# Some helpers for visualization
def get_batch(vocab, model, n_batches=3):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """From Tensorflow's tutorial."""
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
#     plt.savefig(filename)
    plt.show()

tsne = TSNE()
embeds = []
labels = []
for item in get_batch(vocab, model, n_batches=10):
    embeds.append(model[item])
    labels.append(products.loc[int(item)]['product_name'])
embeds = np.array(embeds)

embeds = tsne.fit_transform(embeds)
plot_with_labels(embeds, labels)

# My code
import sklearn as sk
pca = sk.decomposition.PCA(2)
pca.fit(model.wv[vocab])
product2vec = pca.transform(model.wv[vocab])

product2vec = pd.DataFrame(product2vec)
product2vec.columns = ["pca0", "pca1"]
product2vec["product_id"] =

int(vocab)
