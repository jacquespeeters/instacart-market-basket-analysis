# instacart-market-basket-analysis
https://www.kaggle.com/c/instacart-market-basket-analysis

## Intro
Solution i've done which allowed me to place 54th out of 2699 competitors (top2%).

## Official intro

Whether you shop from meticulously planned grocery lists or let whimsy guide your grazing, our unique food rituals define who we are. Instacart, a grocery ordering and delivery app, aims to make it easy to fill your refrigerator and pantry with your personal favorites and staples when you need them. After selecting products through the Instacart app, personal shoppers review your order and do the in-store shopping and delivery for you.

Instacart’s data science team plays a big part in providing this delightful shopping experience. Currently they use transactional data to develop models that predict which products a user will buy again, try for the first time, or add to their cart next during a session. Recently, Instacart open sourced this data - see their blog post on 3 Million Instacart Orders, Open Sourced.

In this competition, Instacart is challenging the Kaggle community to use this anonymized data on customer orders over time to predict which previously purchased products will be in a user’s next order. They’re not only looking for the best model, Instacart’s also looking for machine learning engineers to grow their team.

## How to run
Place data in the data/ folder.

```
python3 create_orders.py
python3 word2vector.py
python3 create_old_orders.py
python3 main.py
```

## Requierements

```
gensim==2.3.0
google-compute-engine==2.4.1
lightgbm==2.0.5
llvmlite==0.19.0
numba==0.34.0
numpy==1.13.1
pandas==0.20.3
scikit-learn==0.18.2
scipy==0.19.1
seaborn==0.8
```

## Main design of the solution

### Strongest feature
My strongest feature is a binary encoding of the purchase pattern of products by users. Binary encoding is a nice way to encode a series of 1/0, isn't it?
``` 
order_prior["UP_order_strike"] = 1 / 2 ** (order_prior["order_number_reverse"])

users_products = order_prior. \
    groupby(["user_id", "product_id"]). \
    agg({..., \
         'UP_order_strike': {"UP_order_strike": "sum"}})
```

### None handling
I created a model specifically to predict None.

The main idea is to re-use feature engineering already done for the main model (like the user profile) and to create new features based on the prediction of the main model. 

Here is the piece of code highlighting the new features created in order to predict None.
```
df_pred = df_pred.groupby(["order_id", "user_id"]). \
    agg({'pred_minus': {'pred_none_prod': "prod"}, \
         'pred': {'pred_basket_sum': "sum", 'pred_basket_std':'std'}})
```
Obviously it is done in a cross-validation way in order to avoid leakage.

It gave me a nice boost +0.1 overall

## Memo perso pour gcloud
Tuto jupyter sur gcloud
https://gist.github.com/valentina-s/79d2443425a921ebfc5e3379ed1ea52a

Commande pour se connecter en ssh sur l'instance avec redirection de port 8888 vers 8888 (permet de rediriger jupyter)
```gcloud compute --project "instacart-175609" ssh --ssh-flag="-L 8888:localhost:8888" --zone "us-central1-c" "instance-1"```
