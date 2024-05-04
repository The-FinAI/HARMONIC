from sklearn.preprocessing import LabelEncoder

from taptap.taptap import Taptap
from taptap.exp_utils import lightgbm_hpo
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import r2_score, f1_score
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def get_score(train_data, val_data, test_data, target_col, task, best_params, use_real_val):
    train_x = train_data.drop(columns=target_col).copy()
    test_x = test_data.drop(columns=target_col).copy()
    train_y = train_data[[target_col]]
    test_y = test_data[[target_col]]
    if use_real_val:
        val_x = val_data.drop(columns=target_col).copy()
        val_y = val_data[[target_col]]
    else:
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=100, random_state=42)
    if task == 'regression':
        gbm = lgb.LGBMRegressor(**best_params)
        gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
        score = r2_score(test_y, pred)
    else:
        gbm = lgb.LGBMClassifier(**best_params)
        gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
        score = f1_score(test_y, pred)
    return score, gbm

# 函数来过滤行
def filter_rows(tabtab_df, real_df, columns):
    filtered_df = tabtab_df.copy()
    for column in columns:
        if column in columns:
            # 从 real_german_train_val 的列中获取唯一值
            valid_values = set(real_df[column].unique())
            # 过滤掉不在 valid_values 集合中的行
            filtered_df = filtered_df[filtered_df[column].isin(valid_values)]
    return filtered_df


if __name__ == '__main__':
    target_col = 'status'
    task = 'classification'
    train_data = pd.read_csv('Data/german/raw/german_train_val.csv')
    test_data = pd.read_csv('Data/german/raw/german_test.csv')
    val_data = pd.read_csv('Data/german/raw/german_val.csv')

    # 使用labelencoder将分类特征转化为数值
    col_name = train_data.columns.tolist()
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

    # 使用一个字典来存储每个列的LabelEncoder
    label_encoders = {}

    for column in columns:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        label_encoders[column] = le

    for column in columns:
        le = LabelEncoder()
        test_data[column] = le.fit_transform(test_data[column])

    for column in columns:
        le = LabelEncoder()
        val_data[column] = le.fit_transform(val_data[column])

    best_params = lightgbm_hpo(
        data=train_data, target_col=target_col, task=task, n_trials=10, n_jobs=16
    )
    original_score, gbm = get_score(
        train_data, val_data, test_data, target_col=target_col, task=task, best_params=best_params, use_real_val=True
    )


    # 解码
    for column in columns:
        train_data[column] = label_encoders[column].inverse_transform(train_data[column].astype(int))

    synthetic_data = pd.read_csv('/share/fengduanyu/SynData/Data/german/syn/tabula_llama_e10_b4_t7.csv')
    synthetic_data = synthetic_data.dropna()
    # 应用过滤函数
    synthetic_data = filter_rows(synthetic_data, train_data, columns)
    if len(synthetic_data) > len(train_data):
        synthetic_data = synthetic_data.head(len(train_data))

    # 使用一个字典来存储每个列的LabelEncoder
    label_encoders2 = {}

    for column in columns:
        le = LabelEncoder()
        synthetic_data[column] = le.fit_transform(synthetic_data[column])
        label_encoders2[column] = le


    # Label generation
    synthetic_data[target_col] = gbm.predict(synthetic_data.drop(columns=[target_col]))

    # Training using synthetic data
    new_score, _ = get_score(
        synthetic_data, val_data, test_data, target_col=target_col, task=task, best_params=best_params, use_real_val=False
    )
    print("The score training by the original data is", original_score)
    print("The score training by the synthetic data is", new_score)

    # 解码
    for column in columns:
        synthetic_data[column] = label_encoders2[column].inverse_transform(synthetic_data[column].astype(int))

    synthetic_data.to_csv('Data/german/syn/taptap+tabula_llama_e10_b4_t7.csv', index=False)







