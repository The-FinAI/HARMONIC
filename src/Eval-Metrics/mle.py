'''这里是用到了验证集，之前的代码没有用到验证集'''
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score, matthews_corrcoef
import xgboost as xgb
from sklearn.model_selection import train_test_split



# def load_data(filename):
#     data = pd.read_csv(filename)
#     X = data.drop(columns=['刑期', '类别'])
#     y = data['类别']
#     z = data['刑期']
#     return X, y, z

def load_data(filename):
    data = pd.read_csv(filename)

    label_encoder = LabelEncoder()
    columns = [
        "Status of existing checking account",
        "Credit history",
        "Purpose",
        "Savings account/bonds",
        "Present employment since",
        "Personal status and sex",
        "Other debtors / guarantors",
        "Property",
        "Other installment plans",
        "Housing",
        "Job",
        "Telephone",
        "foreign worker"
    ]
    for i in columns:
        data[i] = label_encoder.fit_transform(data[i])

    X = data.drop(columns=['status'])
    y = data['status']
    return X, y

# 获取列名
# column_names = data.columns.tolist()

X, y = load_data('/share/fengduanyu/SynData/Data/german/syn/om_e3_b10_t0.7.csv')
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=100, random_state=42)
test_X, test_y = load_data('Data/german/raw/german_test.csv')
# train_X, train_y = load_data('/home/wangyx/relat_to_local/mydata/SynData/raw_data/german_train.csv')
# valid_X, valid_y = load_data('/home/wangyx/relat_to_local/mydata/SynData/raw_data/german_val.csv')
# test_X, test_y = load_data('/home/wangyx/relat_to_local/mydata/SynData/raw_data/german_test.csv')


# # ctgan后再训练
# train_X, train_y,train_z = load_data('../data_balance/kmeans/with_category_train_set_data.csv')
# valid_X, valid_y ,valid_z= load_data('../data_balance/kmeans/with_category_val_set_data.csv')
# test_X, test_y ,test_z= load_data('../data_balance/kmeans/with_category_test_set_data.csv')

# 逻辑回归

parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
# 创建LogisticRegression模型
lr = LogisticRegression(max_iter=1000)
# 使用GridSearchCV来搜索最佳的超参数组合
clf = GridSearchCV(lr, parameters, cv=5)
# 在验证数据上使用GridSearchCV进行拟合，以确定最佳超参数
clf.fit(valid_X, valid_y)

# 输出最佳的超参数组合
best_params = clf.best_params_
print("\nLogistic Best parameters found on validation set: ", best_params)

# 使用最佳的超参数组合在训练集上训练模型
lr_best = LogisticRegression(max_iter=1000, C=best_params['C'], penalty=best_params['penalty'])
lr_best.fit(train_X, train_y)

# 使用训练好的模型在测试集上进行评估
test_pred = lr_best.predict(test_X)
# print("\nLogistic Performance on Test Set:")
# print(classification_report(test_y, test_pred, digits=4))
mcc1 = matthews_corrcoef(test_y, test_pred)
print("Logistic MCC on Test Set:", mcc1)

# 支持向量机
parameters_svm = {'C': [0.5, 1.0], 'kernel': ['linear', 'rbf']}
svm = SVC()
clf_svm = GridSearchCV(svm, parameters_svm, cv=5)
clf_svm.fit(valid_X, valid_y)

best_params_svm = clf_svm.best_params_
print("\nBest parameters for SVM found on validation set: ", best_params_svm)

svm_best = SVC(C=best_params_svm['C'], kernel=best_params_svm['kernel'])
svm_best.fit(train_X, train_y)

test_pred_svm = svm_best.predict(test_X)
# print("\nSVM Performance on Test Set:")
# print(classification_report(test_y, test_pred_svm, digits=4))
mcc2 = matthews_corrcoef(test_y, test_pred_svm)
print("SVM MCC on Test Set:", mcc2)

# 随机森林
parameters_rf = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}

rf = RandomForestClassifier(random_state=42)
clf_rf = GridSearchCV(rf, parameters_rf, cv=5)
clf_rf.fit(valid_X, valid_y)

best_params_rf = clf_rf.best_params_
print("\nBest parameters for RandomForest found on validation set: ", best_params_rf)

rf_best = RandomForestClassifier(n_estimators=best_params_rf['n_estimators'],
                                 max_depth=best_params_rf['max_depth'],
                                 min_samples_split=best_params_rf['min_samples_split'],
                                 random_state=42)
rf_best.fit(train_X, train_y)

test_pred_rf = rf_best.predict(test_X)
# print("\nRandomForest Performance on Test Set:")
# print(classification_report(test_y, test_pred_rf, digits=4))
mcc3 = matthews_corrcoef(test_y, test_pred_rf)
print("RF MCC on Test Set:", mcc3)

# XGB
# parameters_xgb = {'n_estimators': [10, 50], 'learning_rate': [0.01, 0.05], 'max_depth': [3, 5, 7, 10]}
parameters_xgb = {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7, 10]}
xgboost = xgb.XGBClassifier()
clf_xgb = GridSearchCV(xgboost, parameters_xgb, cv=5)
clf_xgb.fit(valid_X, valid_y)

best_params_xgb = clf_xgb.best_params_
print("\nBest parameters for XGBoost found on validation set: ", best_params_xgb)

xgb_best = xgb.XGBClassifier(n_estimators=best_params_xgb['n_estimators'],
                             learning_rate=best_params_xgb['learning_rate'],
                             max_depth=best_params_xgb['max_depth'])
xgb_best.fit(train_X, train_y)

test_pred_xgb = xgb_best.predict(test_X)
# print("\nXGBoost Performance on Test Set:")
# print(classification_report(test_y, test_pred_xgb, digits=4))
mcc4 = matthews_corrcoef(test_y, test_pred_xgb)
print("XGB MCC on Test Set:", mcc4)

value = (mcc1+mcc2+mcc3+mcc4) / 4
print(f'average MCC: {value:.4f}')

# # 使用XGBoost的预测结果作为示例，但你可以针对任何其他模型的预测结果重复此过程
#
# # 获取标签为3且预测错误的索引
# misclassified_idx = test_X[(test_y == 2) & (test_pred_xgb != 2)].index
#
# # 从原始数据集中提取这些数据
# misclassified_data = test_X.loc[misclassified_idx]
#
# # 如果你也想在这个表格中包括真实的标签、预测的标签以及刑期，你可以像这样添加它们：
# misclassified_data['True Label'] = test_y.loc[misclassified_idx]
# misclassified_data['Predicted Label'] = test_pred_xgb[misclassified_idx]
#
# # 从test_X中获取"刑期"并添加到misclassified_data中
# # 假设刑期数据也在test_X中
#
# misclassified_data['刑期'] = test_z.loc[misclassified_idx]
#
# # 将这些数据保存到一个新的CSV文件中
# misclassified_data.to_csv('misclassified_data_label_2.csv', index=False)
def get_model_metrics(true_labels, predicted_labels, model_name):
    """
    Get selected performance metrics of the model in a list.

    Parameters:
    - true_labels: The true labels of the dataset.
    - predicted_labels: The labels predicted by the model.
    - model_name: Name of the model (for display purposes).

    Returns:
    - A list of desired metrics.
    """
    report = classification_report(true_labels, predicted_labels, output_dict=True, digits=4)
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Extract f1-scores for each individual class in order
    class_labels = [label for label in report.keys() if
                    isinstance(label, (int, str)) and label not in ['accuracy', 'macro avg', 'weighted avg']]
    f1_scores_for_classes = [report[label]['f1-score'] for label in sorted(class_labels, key=lambda x: int(x))]

    # Prepare the metrics in the desired order: First the f1-scores for each class, then the general metrics
    metrics = f1_scores_for_classes + [accuracy, report['macro avg']['f1-score'], report['weighted avg']['f1-score']]

    return metrics


# Note: If you want to test this function, you'll need to use it with your data.

# Example usage:
lr_metrics = get_model_metrics(test_y, test_pred, "Logistic Regression")
svm_metrics = get_model_metrics(test_y, test_pred_svm, "Support Vector Machine (SVM)")
rf_metrics = get_model_metrics(test_y, test_pred_rf, "RandomForest")
xgb_metrics = get_model_metrics(test_y, test_pred_xgb, "XGBoost")

print(f"\nLogistic Regression Performance: ", ", ".join([f"{metric:.4f}" for metric in lr_metrics]))
print(f"\nSupport Vector Machine (SVM) Performance: ", ", ".join([f"{metric:.4f}" for metric in svm_metrics]))
print(f"\nRandomForest Performance: ", ", ".join([f"{metric:.4f}" for metric in rf_metrics]))
print(f"\nXGBoost Performance: ", ", ".join([f"{metric:.4f}" for metric in xgb_metrics]))

sum_weighted_f1 = lr_metrics[-1] + svm_metrics[-1] + rf_metrics[-1] + xgb_metrics[-1]
average_weighted_f1 = sum_weighted_f1 / 4
print('average_f1:', f'{average_weighted_f1:.4f}')

