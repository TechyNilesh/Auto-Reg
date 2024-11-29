from river import linear_model, tree, neighbors, ensemble
from river import feature_selection
from river import preprocessing
from river import stats, optim
import numpy as np


def range_gen(min_n,max_n,step=1,float_n=False):
    if float_n:
        return [min_n + i * step for i in range(int((max_n - min_n) / step) + 1)]
    return list(range(min_n,max_n+1,step))

# Regression Config

model_options_reg = [
    linear_model.LinearRegression(),
    tree.HoeffdingAdaptiveTreeRegressor(),
    ensemble.AdaptiveRandomForestRegressor(),
    neighbors.KNNRegressor(),    
    ]

preprocessor_options_reg = [
    preprocessing.MinMaxScaler(),
    preprocessing.MaxAbsScaler(),
    ]

feature_selection_options_reg = [
  feature_selection.SelectKBest(similarity=stats.PearsonCorr())
  ]


hyperparameters_options_reg = {
    'LinearRegression': {
        "l2": range_gen(0.00, 0.01, step=0.001, float_n=True),
        #'optimizer':[optim.Adam(),optim.SGD()]
    },
    'HoeffdingAdaptiveTreeRegressor': {
        'max_depth': range_gen(10,100,step=20),
        'grace_period': range_gen(10,500,step=50),
        "leaf_prediction": ["mean", "model", "adaptive"],
        'bootstrap_sampling': [True,False],
    },
      "AdaptiveRandomForestRegressor": {
        "n_models": range_gen(3,9, step=1),
        "max_depth": range_gen(10, 100, step=10),
        "aggregation_method": ["mean", "median"],
        "grace_period": range_gen(50, 500, step=50),
        "lambda_value": range_gen(2, 10, step=1),
        "tie_threshold": range_gen(0.02, 0.08, step=0.01, float_n=True),
        "leaf_prediction": ["mean", "model", "adaptive"],
    },
    'KNNRegressor':{
        'n_neighbors':range_gen(2,24,step=2),
        'window_size':range_gen(100,1000,step=100),
        'aggregation_method':['mean','median','weighted_mean']
    },
    'MinMaxScaler': {},
    'MaxAbsScaler': {},
    "SelectKBest":{
        "k": range_gen(2,25,step=1),
        #"similarity":[stats.PearsonCorr()],
                  },
}

default_config_dict_reg = {}


default_config_dict_reg['models'] = model_options_reg
default_config_dict_reg['preprocessors'] = preprocessor_options_reg
default_config_dict_reg['features'] = feature_selection_options_reg 
default_config_dict_reg['hyperparameters'] = hyperparameters_options_reg