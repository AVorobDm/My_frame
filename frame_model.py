import pickle
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from abc import ABC, abstractmethod
import os

# from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# import tensorflow as tf
# from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
# import keras
# from keras import backend as K
# from utils import get_activation_by_name
# import utils
# import numpy as np

MAP_SKLEARN_MODEL_NAME_CLASS = {('LR', 'binary'): LogisticRegression,
                        ('LR', 'regression'): Lasso,
                        ('LR', 'multiclass'): LogisticRegression,
                        ('LR', 'multilabel'): LogisticRegression,
                        ('RF', 'binary'): RandomForestClassifier,
                        ('RF', 'regression'): RandomForestRegressor,
                        ('RF', 'multiclass'): RandomForestClassifier}


class UnifiedModelInterface(ABC):
    @abstractmethod
    def __init__(self, objective, model_name, class_num, **kwargs):
        self.objective = objective
        self.class_num = class_num
        pass

    @abstractmethod
    def fit(self, x_train, y_train, x_val, y_val):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def predict_proba(self, x):
        pass

    @abstractmethod
    def save(self, fold_dir):
        pass

    @abstractmethod
    def on_train_end(self):
        pass


class SklearnModel(UnifiedModelInterface):
    def __init__(self, objective, model_name, class_num, **kwargs):
        self.model = MAP_SKLEARN_MODEL_NAME_CLASS[(model_name, objective)](**kwargs)
        self.objective = objective
        self.class_num = class_num
        # self.super().__init__()


    def fit(self, x_train, y_train, x_val, y_val):
        return self.model.fit(x_train, y_train)


    def predict(self, x):
        return self.model.predict(x)


    def predict_proba(self, x):
        if self.objective == 'binary':
            return self.model.predict_proba(x)[:, 1]
        else:
            raise NotImplementedError


    def save(self, fold_dir):
        model_filename = fold_dir + '/model.pkl'
        with open(model_filename, mode='wb') as pickle_dir:
            pickle.dump(self.model, pickle_dir)


    def on_train_end(self):
        del self.model


MAP_BOOST_MODEL_NAME_CLASS = {('LGB', 'binary'): LGBMClassifier,
                        ('LGB', 'regression'): LGBMRegressor,
                        ('CB', 'regression'): CatBoostRegressor,
                        ('CB', 'binary'): CatBoostClassifier}

class GBModel(UnifiedModelInterface):

    def __init__(self, model_name, objective, class_num, **kwargs):
        self.model = MAP_BOOST_MODEL_NAME_CLASS[(model_name, objective)](**kwargs)
        self.objective = objective
        self.class_num = class_num
        # self.super().__init__()

    def fit(self, x_train, y_train, x_val, y_val):
        return self.model.fit(x_train, y_train,
                              eval_set=(x_val, y_val))

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        if self.objective == 'binary':
            return self.model.predict_proba(x)[:, 1]
        else:
            raise NotImplementedError

    def save(self, fold_dir):
        model_filename = fold_dir + '/model.pkl'
        with open(model_filename, mode='wb') as pickle_dir:
            pickle.dump(self.model, pickle_dir)

    def on_train_end(self):
        del self.model

# MAP_NN_MODEL_NAME_CLASS = {('NN', 'binary'): KerasClassifier,
#                            ('NN', 'regression'): KerasRegressor}
#
# class NN(UnifiedModelInterface):
#
#     def __innit__(self, model_name, objective):
#         self.model =  MAP_NN_MODEL_NAME_CLASS[(model_name,
#                                                objective)](base_fn())
#
#     def base_fn(self, input_size, ):
#         mod = keras.Sequential()
#         mod.add(keras.Dense(units, input_size, activation))
#         mod.add(keras.Dense(units/2, input_size, activation))
#         if objective == 'binary':
#             mod.add(keras.Dense(1, ))
#         elif objective == 'multiclass':
