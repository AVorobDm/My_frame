import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


class Preprocessor:
    def __init__(self, fill_na_type=None, cat_features=None):
        if cat_features is not None:
            self.cat_features = cat_features
            self.enc = None

        if fill_na_type is not None:
            self.fill_na_type = fill_na_type

    def fill_na(self, x):
        if self.fill_na_type == 'mean':
            filled_x = x.fillna(x.mean())
        elif self.fill_na_type == 'median':
            filled_x = x.fillna(np.median(x))
        elif isinstance(self.fill_na_type, int):
            filled_x = x.fillna(self.fill_na_type)
        else:
            raise NotImplementedError
        return filled_x

    def encode(self, x, fit=True):
        if self.cat_features is None:
            return x

        if fit:
            self.enc = OneHotEncoder()
            self.enc.fit(x[self.cat_features])

        cat_one_hotted = pd.DataFrame(self.enc.transform(x).toarray(),
                                      index=x.index,
                                      columns=self.enc.get_feature_names())
        x_no_cat = x.drop(self.cat_features, axis=1)

        final_x = pd.concat([x_no_cat, cat_one_hotted], axis=1)

        return final_x


