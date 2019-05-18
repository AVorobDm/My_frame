from frame_model import GBModel, SklearnModel #MLP
import pandas as pd
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def test_code(train_df, test_df, y_true, name=None, obj=None,
              model_type=None, class_num=None, pred_type=None):
    cv_preds = []
    ans = []
    pre = Preprocessor(scaling='StandardScaler', split='KFold', fillnaer='mean', k=4)
    if model_type == 'sklearn':
        model = SklearnModel(obj, name, class_num)
    elif model_type == 'gbm':
        model = GBModel(obj, name, class_num)
    elif model_type == 'nn':
        model = MLP(obj, name, class_num, nn_params)

    train_data = pd.DataFrame(pre.get_scaled(pre.small_prep(train_df)), index=train_df.index, columns=train_df.columns)
    test_data = pd.DataFrame(pre.get_scaled(pre.small_prep(test_df)), index=test_df.index, columns=test_df.columns)

    if model_type == 'nn':
        model._build(units=(100, 100), kernel_initializer='glorot_uniform', activation='relu',
                     dropout=0.3, opty='Adam')

    cv_list = pre.get_splits(train_data, y_true)

    for i in cv_list:
        model.fit(train_data.iloc[i[0]], y_true.iloc[i[0]], train_data.iloc[i[1]], y_true.iloc[i[1]])
        cv_pred = model.predict(train_data.iloc[i[1]])
        cv_preds.append(mean_squared_error(y_true.iloc[i[1]], cv_pred))

        print(cv_preds)

    if pred_type == 'proba':
        ans = model.predict_proba(test_data)
    else:
        ans = model.predict(test_data)

    model.on_train_end()
    return ans


train_df = pd.read_csv(r'C:\Users\vpd_l\Desktop\Frame\train_houses.csv', index_col='Id')
test_df = pd.read_csv(r'C:\Users\vpd_l\Desktop\Frame\test_houses.csv', index_col='Id')
y_true = np.log(train_df.SalePrice)

train_df.drop(['SalePrice', 'MSSubClass', 'MSZoning', 'LotFrontage', 'Street', 'Alley',
            'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
            'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
            'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
            'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], axis=1, inplace=True)

test_df.drop(['MSSubClass', 'MSZoning', 'LotFrontage', 'Street', 'Alley',
            'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
            'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
            'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
            'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], axis=1, inplace=True)

nn_params = {'early_stopping_rounds': 200,
            'reduce_lr_patience':20,
            # 'reduce_lr_factor': ,
            'verbose': 1,
            'batch_size': 32,
            'epochs': 20,
            'monitor': 'val_accuracy',
            'monitor_mode': 'max'}
            # 'class_weight': ,
            # 'cp_path': }

pred = test_code(train_df, test_df, y_true, 'CB', 'regression', 'gbm', 1)

pd.DataFrame({'SalePrice': np.exp(pred)}, index=test_df.index).to_csv(r'C:\Users\vpd_l\Desktop\Frame\houses_catboost_predict.csv')
