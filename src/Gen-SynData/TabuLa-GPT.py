from tabula_raw import Tabula
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import random
import json

def filter_rows(syn_df, real_df, columns):
    filtered_df = syn_df.copy()
    for column in columns:
        # 从 real_german_train_val 的列中获取唯一值
        valid_values = set(real_df[column].unique())
        # 过滤掉不在 valid_values 集合中的行
        filtered_df = filtered_df[filtered_df[column].isin(valid_values)] #.astype(int)
        if len(filtered_df) == 0:
            return filtered_df
    return filtered_df

data = pd.read_csv("Data/german/raw/german_train_val.csv")

# 使用labelencoder将分类特征转化为数值
col_name = data.columns.tolist()
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
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Rename the columns based on the suggested abbreviations
data.columns = [
    "CheckingStatus", "Duration", "CreditHist", "Purpose", "CreditAmt",
    "Savings", "EmploySince", "InstallRate", "PersStatus", "Debtors",
    "ResidSince", "Property", "Age", "InstallPlans", "Housing",
    "BankCredits", "Job", "MaintLiable", "Telephone", "ForeignWorker", "Status"
]

cat_columns = [
    "CheckingStatus",
    "CreditHist",
    "Purpose",
    "Savings",
    "EmploySince",
    "PersStatus",
    "Debtors",
    "Property",
    "InstallPlans",
    "Housing",
    "Job",
    "Telephone",
    "ForeignWorker"
]

model = Tabula(llm="Base-Models/distilgpt2", experiment_dir="results/FT-CP/German-TabuLa-GPT2_e10_b1_t7",
               batch_size=1, epochs=10, efficient_finetuning="lora")

# model.model.load_state_dict(torch.load("pretrained-model/model.pt"), strict=False)

model.fit(data)

# save = torch.save(model.model.state_dict(), "results/FT-LLMs/German-TabuLa-LLaMA2-Chat/model_e3_b10_t7.pt")

synthetic_data = model.sample(n_samples=len(data), temperature=0.7, max_length=1024)
synthetic_data = filter_rows(synthetic_data, data, cat_columns)

while len(synthetic_data) < len(data):
    new_data = len(data)-len(synthetic_data)
    print(f"还需生成{new_data}条样本")
    new_syn_data = model.sample(n_samples=new_data, temperature=0.7, max_length=1024)
    new_syn_data = filter_rows(new_syn_data, data, cat_columns)
    synthetic_data = pd.concat([synthetic_data, new_syn_data], ignore_index=True)
# 解码
synthetic_data.columns = col_name
for column in columns:
    synthetic_data[column] = label_encoders[column].inverse_transform(synthetic_data[column].astype(int))
synthetic_data.to_csv("Data/german/syn/tabula_gpt_e10_b1_t7.csv", index=False)