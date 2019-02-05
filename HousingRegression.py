import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso


def prepare_data():
    global boston_market_data, boston_data, boston_data_full
    boston_market_data = load_boston()
    boston_data = pd.DataFrame(boston_market_data.data, columns = [boston_market_data.feature_names])
    boston_data_full = pd.DataFrame(boston_market_data.data, columns = [boston_market_data.feature_names])

def preprocessing_data():
    global target_cut, target_cut_full, scaled_data, scaled_data_full
    target_cut = np.array(boston_market_data['target'])
    target_cut_full = np.array(boston_market_data['target'])
    k = boston_data.values
    k = k.T

    delete_max = set()
    delete_min = set()
    g = {0, 5, 6, 11, 12}
    for x in g:
        a = max(k[x])
        for i in range(k.shape[1]):
         if a == k[x][i]:
              delete_max.add(i)

    g = {0, 5, 6, 11, 12}
    for x in g:
        a = min(k[x])
        for i in range(k.shape[1]):
            if a == k[x][i]:
                delete_min.add(i)

    for i in range(len(delete_max)):
        boston_data.drop(boston_data.index[max(delete_max)], inplace = True)
        target_cut = np.delete(target_cut,max(delete_max))
        delete_max.remove(max(delete_max))

    for i in range(len(delete_min)):
        if(max(delete_min) < boston_data.shape[0]):
            boston_data.drop(boston_data.index[max(delete_min)], inplace = True)
            target_cut=np.delete(target_cut,max(delete_min))
        delete_min.remove(max(delete_min))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(boston_data)
    scaled_data_full = scaler.fit_transform(boston_data_full)

def lasso_regression():
    lasso_regression = Lasso(alpha = 0.6)
    lasso_regression.fit(boston_train_data, boston_train_target)
    score = lasso_regression.score(boston_test_data_full, boston_test_target_full)
    print("Lasso regression variance score: %.2f" % score)
    print("CrossValidation")
    scores = cross_val_score(Lasso(), scaled_data_full, target_cut_full, cv = 8)
    print(scores)

prepare_data()
preprocessing_data()
boston_train_data, boston_test_data, boston_train_target,boston_test_target = train_test_split(scaled_data, target_cut, test_size = 0.1)
boston_train_data_full, boston_test_data_full, boston_train_target_full,boston_test_target_full = train_test_split(scaled_data_full, target_cut_full, test_size = 0.1)
lasso_regression()

