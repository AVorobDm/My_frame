
# coding: utf-8

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold


# In[ ]:


class Preprocessor():
    
    def __init__(self, scaling=None, split=None, k=3):
        self.scaling = scaling
        self.split = split
        self.k = k
        
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

