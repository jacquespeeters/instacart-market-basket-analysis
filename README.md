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

## Memo perso pour gcloud
Tuto jupyter sur gcloud
https://gist.github.com/valentina-s/79d2443425a921ebfc5e3379ed1ea52a

Commande pour se connecter en ssh sur l'instance avec redirection de port 8888 vers 8888 (permet de rediriger jupyter)
```gcloud compute --project "instacart-175609" ssh --ssh-flag="-L 8888:localhost:8888" --zone "us-central1-c" "instance-1"```
