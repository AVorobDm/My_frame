import pickle
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from abc import ABC, abstractmethod
import os

import tensorflow as tf
import keras
from keras import backend as K
from utils import get_activation_by_name
import utils
import numpy as np


class UnifiedModelInterface(ABC):

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

def get_activation_by_name(activation_name):
    if activation_name == 'leaky_relu':
        return keras.layers.LeakyReLU()
    elif activation_name == 'prelu':
        return keras.layers.PReLU()
    else:
        return keras.layers.Activation(activation_name)


optimizers_MAP = {'Adagrad': keras.optimizers.Adagrad,
                  'SGD': keras.optimizers.SGD,
                  'Adam': keras.optimizers.Adam,
                  'Nadam': keras.optimizers.Nadam,
                  'RMSprop': keras.optimizers.RMSprop,
                  'Adadelta': keras.optimizers.Adadelta
                 }



class MlpModel(UnifiedModelInterface):
    def __init__(self, objective, _, class_num, **kwargs):
        if objective == 'binary':
            self.loss = 'binary_crossentropy'
            self.metric = ['accuracy']
            self.ouput_num = 1
            self.output_activation = 'sigmoid'
        elif objective == 'multilabel':
            self.loss = 'binary_crossentropy'
            self.metric = ['accuracy']
            self.ouput_num = class_num
            self.output_activation = 'sigmoid'
        elif objective == 'multiclass':
            self.loss = 'categorical_crossentropy'
            self.metric = ['accuracy']
            self.ouput_num = class_num
            self.output_activation = 'softmax'
	elif objective == 'regression':
            self.loss = 'mean_squared_error'
            self.metric = ['rmse']
            self.ouput_num = 1
            self.output_activation = 'linear'
        else:
            raise NotImplementedError

        self.model = self._build(**kwargs)

        self.objective = objective

        self.early_stopping_rounds = kwargs['early_stopping_rounds']
        self.reduce_lr_patience = kwargs['reduce_lr_patience']
        self.reduce_lr_factor = kwargs['reduce_lr_factor']
        self.verbose = kwargs['verbose']
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']
        self.monitor = kwargs['monitor']
        self.monitor_mode = kwargs['monitor_mode']
        self.class_weight = kwargs['class_weight']
        self.cp_path = kwargs['cp_path']


    def _build(self, units= (100,100), optimizer='Adam', l2=0,
               learning_rate=0.001, dropout=0.5,
               activations='relu', batch_norm=False,
               kernel_initializer='glorot_uniform', *args,**kwargs
               ):
        model = keras.models.Sequential()

        for neuron_num in units:
            model.add(keras.layers.Dense(neuron_num,
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=
                                         keras.regularizers.l2(l2)
                                         )
                      )

            model.add(get_activation_by_name(activations))

            if batch_norm:
                model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(dropout))

        model.add(keras.layers.Dense(self.ouput_num,
                                     kernel_initializer=kernel_initializer))

        model.add(get_activation_by_name(self.output_activation))
        model.compile(loss=self.loss, metrics=self.metric,
                      optimizer=utils.optimizers_MAP[optimizer](lr=
                                                                learning_rate))

        return model


    def fit(self, x_train, y_train, x_val, y_val):
        """

        Args:
            x_train: training set of features
            y_train: training set of labels
            x_val: validation set of features
            y_val: validations set of labels

        """

        early_stopping = keras.callbacks.EarlyStopping(monitor=self.monitor,
            patience=self.early_stopping_rounds, verbose=self.verbose,
            mode=self.monitor_mode, restore_best_weights=True)

        reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor=self.monitor,
            factor=self.reduce_lr_factor, patience=self.reduce_lr_patience,
            verbose=self.verbose, min_delta=1e-4, mode=self.monitor_mode)

        self.model.fit(x=x_train.values, y=y_train.values,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=self.verbose,
                       callbacks=[reduce_lr_loss,early_stopping],
                       validation_data=(x_val.values, y_val.values),
                       shuffle=True
                       )


    def predict(self, feat_df):
        if self.objective == 'binary':
            pred = self.model.predict(feat_df.values).flatten().round()
	elif self.objective == 'regression':
	    pred = self.model.predict(feat_df.values).flatten()
        elif self.objective == 'multiclass':
            pred = self.model.predict(feat_df.values).argmax(axis=1)
	else:
	    pred = self.model.predict(feat_df.values)
	return pred


    def predict_proba(self, feat_df):
        if self.objective == 'regression':
            raise ValueError
        elif self.objective == 'binary':
            pred = self.model.predict(feat_df.values).flatten()
	else:
            pred = self.model.predict(feat_df.values)
        return pred


    def save(self, fold_dir):
        self.model.model.save(os.path.join(fold_dir,'model.h5'))


    def on_train_end(self):
        del self.model
        K.clear_session()
