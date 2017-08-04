from ggplot import *
import utils


param_grid = {'objective': ['binary']
    , 'metric': ['binary_logloss']
    , 'learning_rate': [0.9]
    , 'verbose': [0]
              ,'num_leaves': [5, 10]}

data = utils.grid_search(lgb_train, lgb_valid, param_grid)

print(data)

ggplot(aes(x='num_leaves', y='best_score'), data=data) + geom_point() + geom_line()
