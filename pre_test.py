from preprocessor import Preprocessor
import pandas as pd

train = pd.read_csv(r'C:\Users\vpd_l\Desktop\Frame\train_titanick.csv', index_col='PassengerId')

num_cols = [
    'Age',
    'Fare'
]

cat_cols = [
    'Pclass',
    'Name',
    'Sex',
    'SibSp',
    'Parch',
    'Ticket',
    'Cabin',
]

pre = Preprocessor(scaling='StandardScaler', split='KFold', fillnaer='mean', k=3, cats=cat_cols, nums=num_cols)

train_df = pd.DataFrame(pre.get_scaled(pre.small_prep(train[num_cols])), index=train.index, columns=num_cols)
ohe_df = pre.ohe(train, fit=True)

final_df = pre.concut(ohe_df, train_df)

final_df.to_csv('try_titanick.csv')
