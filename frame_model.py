import pickle
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from abc import ABC, abstractmethod
import os

# from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

import tensorflow as tf
import keras
from keras import backend as K
import numpy as np

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
        super(SklearnModel, self).__init__(objective, model_name, class_num)
        self.model = MAP_SKLEARN_MODEL_NAME_CLASS[(model_name, objective)](**kwargs)

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

    def __init__(self, objective, model_name, class_num, **kwargs):
        super(GBModel, self).__init__(objective, model_name, class_num)
        self.model = MAP_BOOST_MODEL_NAME_CLASS[(model_name, objective)](**kwargs)

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


def get_activation_by_name(activation=None):
    if activation == 'leaky_relu':
        return keras.layers.LeakyReLU()
    elif activation == 'prelu':
        return keras.layers.PReLU()
    else:
        return keras.layers.Activation(activation)

OPT_MAP = {'Adagrad': keras.optimizers.Adagrad,
                  'SGD': keras.optimizers.SGD,
                  'Adam': keras.optimizers.Adam,
                  'Nadam': keras.optimizers.Nadam,
                  'RMSprop': keras.optimizers.RMSprop,
                  'Adadelta': keras.optimizers.Adadelta}


class MLP(UnifiedModelInterface):
    def __init__(self, objective, _, class_num, **kwargs):
        if objective == 'binary':
            self.loss = 'binary_crossentropy'
            self.metric = ['accuracy']
            self.output_num = 1
            self.last_act = 'sigmoid'
        elif objective == 'multiclass':
            self.loss = 'categorical_crossentropy'
            self.metric = ['accuracy']
            self.output_num = class_num
            self.last_act = 'softmax'
        elif objective == 'multilabel':
            self.loss = 'binary_crossentropy'
            self.metric = ['accuracy']
            self.output_num = class_num
            self.last_act = 'sigmoid'
        elif objective == 'regression':
            self.loss = 'mean_squared_error'
            self.metric = ['mse']
            self.output_num = 1
            self.last_act = 'linear'
        else:
            raise NotImplementedError

        self.model = self._build(**kwargs)

        self.objective = objective

        self.early_stopping_rounds = kwargs['early_stopping_rounds']
        self.reduce_lr_patience = kwargs['reduce_lr_patience']
        # self.reduce_lr_factor = kwargs['reduce_lr_factor']
        self.verbose = kwargs['verbose']
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']
        self.monitor = kwargs['monitor']
        self.monitor_mode = kwargs['monitor_mode']
        # self.class_weight = kwargs['class_weight']
        # self.cp_path = kwargs['cp_path']


    def _build(self, units=(100,100), kernel_initializer=None, l2=0, activation=None,
               batch_norm=False, dropout=0.5, opty=None, learning_rate=0.01, *args,**kwargs):
        model = keras.models.Sequential()

        for num_units in units:
            model.add(keras.layers.Dense(num_units, kernel_initializer=kernel_initializer,
                                         kernel_regularizer=keras.regularizers.l2(l2)))
            model.add(get_activation_by_name(activation))

            if batch_norm:
                model.add(keras.layers.BatchNormalization())

            model.add(keras.layers.Dropout(dropout))

        model.add(keras.layers.Dense(self.output_num,
                                     kernel_initializer=kernel_initializer))
        model.add(get_activation_by_name(self.last_act))

        model.compile(loss=self.loss, optimizer=OPT_MAP[opty](lr=learning_rate), metrics=self.metric)

        return model


    def fit(self, x_train, y_train, x_val, y_val):
        stopper = keras.callbacks.EarlyStopping(monitor=self.monitor,
                                                patience=self.early_stopping_rounds,
                                                verbose=self.verbose,
                                                mode=self.monitor_mode,
                                                baseline=None,
                                                restore_best_weights=True)

        self.model.fit(x=x_train.values, y=y_train.values,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=self.verbose,
                       callbacks=[stopper],
                       validation_data=(x_val.values, y_val.values),
                       shuffle=True)

    def predict(self, X):
        if self.objective == 'binary':
            return self.model.predict(X.values).flatten().round()
        elif self.objective == 'regression':
            return self.model.predict(X.values).flatten()
        elif self.objective == 'multiclass':
            return self.model.predict(X.values).argmax(axis=1)
        else:
            return self.model.predict(X.values)


    def predict_proba(self, X):
        if self.objective == 'regression':
            raise ValueError
        if self.objective == 'binary':
            return self.model.predict(X.values).flatten()
        else:
            return self.model.predict(X.values)

    def save(self, fold_dir):
        self.model.model.save(os.path.join(fold_dir, 'model.h5'))


    def on_train_end(self):
        del self.model
        K.clear_session()

