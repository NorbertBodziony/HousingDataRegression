import numpy as np
from sklearn.datasets import load_boston
boston_market_data = load_boston()
#print(boston_market_data['DESCR'])

print(boston_market_data['data'].shape)
print(boston_market_data['target'].shape)



import pandas as pd
boston_data = pd.DataFrame(boston_market_data.data, columns=[boston_market_data.feature_names])
#print(boston_data.head())

Target_cut=np.array(boston_market_data['target'])
k=boston_data.values
k=k.T
print(type(boston_market_data))
Delete=set()
for x in range(k.shape[0]):
    a=max(k[x])
    for i in range(k.shape[1]):
     if a==k[x][i]:
          Delete.add(i)

for i in range(len(Delete)):
    boston_data.drop(boston_data.index[max(Delete)], inplace=True)
    Target_cut=np.delete(Target_cut,max(Delete))
    Delete.remove(max(Delete))

print(Target_cut.shape)
#print(boston_market_data['target'])
print(boston_data.shape)
#print(boston_data.iloc[1]["CRIM"])

#print(max(boston_data["CRIM"]))
import matplotlib.pyplot as plt
import seaborn as sb

#print(sb.pairplot(boston_data, diag_kind="kde"))


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(boston_data)

from sklearn.model_selection import train_test_split

boston_train_data, boston_test_data, \
boston_train_target,boston_test_target = \
train_test_split(scaled_data,Target_cut, test_size=0.1)

# print("Training dataset:")
# print("patients_train_data:", boston_train_data.shape)
# print("patients_train_target:", boston_train_target.shape)
#
# print("Testing dataset:")
# print("patients_test_data:", boston_test_data.shape)
# print("patients_test_target:", boston_test_target.shape)
#
#
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(boston_train_data, boston_train_target)

id=1
linear_regression_prediction = linear_regression.predict(boston_test_data[id,:].reshape(1,-1))
#
# print("Model predicted for patient {0} value {1}".format(id, linear_regression_prediction))
# print("Real value for patient \"{0}\" is {1}".format(id, boston_test_target[id]))

from sklearn.metrics import mean_squared_error
print("Mean squared error of a learned model: %.2f" %
      mean_squared_error(boston_test_target, linear_regression.predict(boston_test_data)))

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(boston_test_target, linear_regression.predict(boston_test_data)))


print("CrossValidation")
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(), scaled_data, Target_cut, cv=8)
print(scores)

from sklearn.linear_model import Lasso
lasso_regression = Lasso(alpha=0.6)
lasso_regression.fit(boston_train_data, boston_train_target)
score = lasso_regression.score(boston_test_data, boston_test_target) #r2 score
print("Lasso regression variance score: %.2f" % score)

print('Coefficients of a learned model: \n', lasso_regression.coef_)