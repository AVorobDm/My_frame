from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

class Preprocessor():
    
    def __init__(self, scaling=None, split=None, fillnaer=None, k=3):
        self.scaling = scaling
        self.split = split
        self.k = k
        # self.cats = cats
        # self.nums = nums

        if fillnaer is not None:
            self.fillnaer = fillnaer

    def small_prep(self, X):

        if self.fillnaer == 'mean':
            filled_X = X.fillna(np.mean(X))
        elif self.fillnaer == 'median':
            filled_X = X.fillna(np.median(X))
        elif type(self.fillnaer) == int:
            filled_X = X.fillna(self.fillnaer)
        else:
            raise NotImplementedError
        return filled_X


    # def ohe(self, X, fit=True):
    #     if self.cats is None:
    #         return X
    #
    #     if fit:
    #         self.enc = OneHotEncoder()
    #         self.enc.fit(X[self.cats])
    #
    #     cat_one_hotted = pd.DataFrame(self.enc.transform(X).toarray(),
    #                                   index=X.index,
    #                                   columns=self.enc.get_feature_names())
    #     return cat_one_hotted


    def get_scaled(self, X):
        if self.scaling == 'MinMaxScaler':
            scaler = MinMaxScaler()
            self.scaled_X = scaler.fit_transform(X)
            
        elif self.scaling == 'StandardScaler':
            scaler = StandardScaler()
            self.scaled_X = scaler.fit_transform(X)
            
        else:
            self.scaled_X = X
        return self.scaled_X


    # def concut(self, X_cats, X_num):
    #     final_X = pd.concat([X_cats, X_num], axis=1)
    #
    #     return final_X
            
        
    def get_splits(self, X, y=None):
        if self.split == 'KFold':
            kf = KFold(n_splits=self.k)
            splited = kf.split(X) 
            
        elif self.split == 'StratifiedKFold':
            skf = StratifiedKFold(n_splits=self.k)
            splited = skf.split(X,y)
            
        else:
            raise ValueError
            
        return list(splited)

