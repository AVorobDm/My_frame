from frame_model import GBModel
import pandas as pd
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def test_code(train_df, test_df, y_true, pred_type='None'):
    cv_preds = []
    ans = []
    preprocessor = Preprocessor(scaling='StandardScaler', split='KFold', k=4)
    model = GBModel('LGB', 'regression', 1)
    train_data = pd.DataFrame(preprocessor.get_scaled(train_df), index=train_df.index, columns=train_df.columns)
    test_data = pd.DataFrame(preprocessor.get_scaled(test_df), index=test_df.index, columns=test_df.columns)
    cv_list = preprocessor.get_splits(train_data, y_true)

    for i in cv_list:
        model.fit(train_data.iloc[i[0]], y_true.iloc[i[0]], train_data.iloc[i[1]], y_true.iloc[i[1]])
        cv_pred = model.predict(train_data.iloc[i[1]])
        cv_preds.append(mean_squared_error(y_true.iloc[i[1]], cv_pred))

    print(cv_preds)

    if pred_type == 'proba':
        ans = model.predict_proba(test_data)
    else:
        ans = model.predict(test_data)

    return ans


train_df = pd.read_csv(r'C:\Users\vpd_l\Desktop\Frame\train_titanick.csv', index_col='Id')
test_df = pd.read_csv(r'C:\Users\vpd_l\Desktop\Frame\test_titanick.csv', index_col='Id')
y_true = train_df.winPlacePerc
train_df.drop(['winPlacePerc', 'matchType', 'groupId', 'matchId'], axis=1, inplace=True)
test_df.drop(['matchType', 'groupId', 'matchId'], axis=1, inplace=True)

pred = test_code(train_df, test_df, y_true)

pd.DataFrame({'winPlacePerc': pred}, index=test_df.index).to_csv(r'C:\Users\vpd_l\Desktop\Frame\kaggle_V2_predict.csv')
