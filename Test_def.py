import pandas as pd
from frame_model import GBModel, SklearnModel, MLP
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def test_code(train_df, test_df, y_true, name=None, obj=None,
              model_type=None, class_num=None, pred_type=None):
    cv_preds = []
    ans = []
    pre = Preprocessor(scaling='StandardScaler', split='KFold', fillnaer='mean', k=2)
    if model_type == 'sklearn':
        model = SklearnModel(obj, name, class_num)
    elif model_type == 'gbm':
        model = GBModel(obj, name, class_num)
    elif model_type == 'nn':
        model = MLP(obj, name, class_num, **nn_params)

    train_data = pd.DataFrame(pre.get_scaled(pre.small_prep(train_df)), index=train_df.index, columns=train_df.columns)
    test_data = pd.DataFrame(pre.get_scaled(pre.small_prep(test_df)), index=test_df.index, columns=test_df.columns)

    cv_list = pre.get_splits(train_data, y_true)

    for i in cv_list:
        model.fit(train_data.iloc[i[0]], y_true.iloc[i[0]], train_data.iloc[i[1]], y_true.iloc[i[1]])
        cv_pred = model.predict(train_data.iloc[i[1]])
        cv_preds.append(np.log(mean_squared_error(y_true.iloc[i[1]], cv_pred)))

    print(cv_preds)

    if pred_type == 'proba':
        ans = model.predict_proba(test_data)
    else:
        ans = model.predict(test_data)

    model.on_train_end()
    return ans

nn_params = {'early_stopping_rounds': 20,
            'reduce_lr_patience':20,
            'reduce_lr_factor': 0.1,
            'verbose': 1,
            'batch_size': 64,
            'epochs': 200,
            'monitor': 'val_loss',
            'monitor_mode': 'min',
            'units': (100, 100),
            'kernel_initializer': 'glorot_uniform',
            'l2': 0,
            'activation': 'relu',
            'batch_norm': False,
            'dropout': 0.5,
            'opty': 'Adam',
            'learning_rate': 0.01}