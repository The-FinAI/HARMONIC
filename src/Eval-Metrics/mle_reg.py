'''这里是用到了验证集，之前的代码没有用到验证集'''
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score, matthews_corrcoef
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import train_test_split


def load_data(filename):
    data = pd.read_csv(filename)
    X = data.drop(columns=['刑期'])
    y = data['刑期']
    return X, y

# 获取列名
# column_names = data.columns.tolist()

# 获取数据
# X, y = load_data('../fraud/CTAB+/Fraud_fake_19.csv')
# train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.12, random_state=42)
# test_X, test_y = load_data('../datanew/real_fraud_test.csv')
train_X, train_y = load_data('/home/wangyx/relat_to_local/mydata/SynData/raw-data/fraud/jzh_fraud_train.csv')
valid_X, valid_y = load_data('/home/wangyx/relat_to_local/mydata/SynData/raw-data/fraud/jzh_fraud_val.csv')
test_X, test_y = load_data('/home/wangyx/relat_to_local/mydata/SynData/raw-data/fraud/jzh_fraud_test.csv')


# 支持向量机
parameters_svm = {'C': [0.5, 1.0], 'kernel': ['linear', 'rbf']}
svm = SVR()
reg_svm = GridSearchCV(svm, parameters_svm, cv=5)
reg_svm.fit(valid_X, valid_y)

best_params_svm = reg_svm.best_params_
print("\nBest parameters for SVM found on validation set: ", best_params_svm)

svm_best = SVR(C=best_params_svm['C'], kernel=best_params_svm['kernel'])
svm_best.fit(train_X, train_y)

test_pred_svm = svm_best.predict(test_X)
r2_svm = r2_score(test_y, test_pred_svm)
print("\nSVM Performance on Test Set:", r2_svm)

# 随机森林
parameters_rf = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}

rf = RandomForestRegressor(random_state=42)
reg_rf = GridSearchCV(rf, parameters_rf, cv=5)
reg_rf.fit(valid_X, valid_y)

best_params_rf = reg_rf.best_params_
print("\nBest parameters for RandomForest found on validation set: ", best_params_rf)

rf_best = RandomForestRegressor(n_estimators=best_params_rf['n_estimators'],
                                 max_depth=best_params_rf['max_depth'],
                                 min_samples_split=best_params_rf['min_samples_split'],
                                 random_state=42)
rf_best.fit(train_X, train_y)

test_pred_rf = rf_best.predict(test_X)
r2_rf = r2_score(test_y, test_pred_rf)
print("\nRandomForest Performance on Test Set:", r2_rf)

# XGB
# parameters_xgb = {'n_estimators': [10, 50], 'learning_rate': [0.01, 0.05], 'max_depth': [3, 5, 7, 10]}
parameters_xgb = {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7, 10]}
xgboost = xgb.XGBRegressor()
reg_xgb = GridSearchCV(xgboost, parameters_xgb, cv=5)
reg_xgb.fit(valid_X, valid_y)

best_params_xgb = reg_xgb.best_params_
print("\nBest parameters for XGBoost found on validation set: ", best_params_xgb)

xgb_best = xgb.XGBRegressor(n_estimators=best_params_xgb['n_estimators'],
                             learning_rate=best_params_xgb['learning_rate'],
                             max_depth=best_params_xgb['max_depth'])
xgb_best.fit(train_X, train_y)

test_pred_xgb = xgb_best.predict(test_X)
r2_xgb = r2_score(test_y, test_pred_xgb)
print("\nXGBoost Performance on Test Set:", r2_xgb)

# MLP
parameters_mlp = {'hidden_layer_sizes':[(50,), (100,), (150,)], 'activation':['relu', 'sigmoid'], 'solver':['adam'], 'max_iter':[500]}
# Creating the MLP Regressor model
# mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
mlp = MLPRegressor()
reg_mlp = GridSearchCV(mlp, parameters_mlp, cv=5)
reg_mlp.fit(valid_X, valid_y)

best_params_mlp = reg_mlp.best_params_
print("\nBest parameters for MLP found on validation set: ", best_params_mlp)

mlp_best = MLPRegressor(hidden_layer_sizes=best_params_mlp['hidden_layer_sizes'],
                        activation=best_params_mlp['activation'],
                        solver=best_params_mlp['solver'],
                        max_iter=best_params_mlp['max_iter'])
mlp_best.fit(train_X, train_y)
# Training the model
# mlp_regressor.fit(train_X, train_y)

# Predicting on validation set
test_pred_mlp = mlp_best.predict(test_X)
r2_mlp = r2_score(test_y, test_pred_mlp)
print("\nMLP Performance on Test Set:", r2_mlp)
# Calculating Mean Squared Error (MSE)
# mse = mean_squared_error(valid_y, test_pred_mlp)
# rmse = np.sqrt(mse)


# Note: If you want to test this function, you'll need to use it with your data.

# Example usage:

sum_r2 = r2_svm + r2_rf + r2_xgb + r2_mlp
average_r2 = sum_r2 / 4
print('average_r2:', f'{average_r2:.4f}')

