from tabula import Tabula
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

def ana_conditon(syn_data):
    if len(syn_data) < samples:
        return True
    elif syn_data.isna().any().any():
        return True
    # elif not all(str(element).isdigit() for element in syn_data.loc[0, num_columns]):
    #     return True
    else:
        return False

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

num_columns = [
    "Duration",
    "CreditAmt",
    "InstallRate",
    "ResidSince",
    "Age",
    "BankCredits",
    "MaintLiable"
]

model = Tabula(llm="Base-Models/llama-2-7b-chat-T", experiment_dir="results/FT-CP/German-TabuLa-LLaMA2-Chat_e10_b4",
               batch_size=4, epochs=10, efficient_finetuning="lora")

# model.model.load_state_dict(torch.load("pretrained-model/model.pt"), strict=False)

model.fit(data)

save = torch.save(model.model.state_dict(), "results/FT-LLMs/German-TabuLa-LLaMA2-Chat/model_e10_b4_t7.pt")

samples = 800

synthetic_data = model.sample(n_samples=samples, temperature=0.7, max_length=1024)
synthetic_data = filter_rows(synthetic_data, data, cat_columns)

while len(synthetic_data) < samples:
    print(f"还需生成{samples-len(synthetic_data)}条样本")
    new_syn_data = model.sample(n_samples=samples-len(synthetic_data), temperature=0.7, max_length=1024)
    new_syn_data = filter_rows(new_syn_data, data, cat_columns)
    synthetic_data = pd.concat([synthetic_data, new_syn_data], ignore_index=True)
# synthetic_df = pd.DataFrame(columns=data.columns.tolist())
# for index in range(10):
#     # starting_prompts = choose_begining(data, index)
#     synthetic_data = model.sample(n_samples=1, temperature=0.7, max_length=1024)
#     synthetic_data = filter_rows(synthetic_data, data, cat_columns)
#     # 判断是否存在某些列生成None值，如果存在，则重新生成
#     cycle = 0
#     # or synthetic_data.isna().any().any()
#     while ana_conditon(synthetic_data) and cycle <= 20:
#         synthetic_data = model.sample(n_samples=1, temperature=0.7, max_length=1024)
#         synthetic_data = filter_rows(synthetic_data, data, cat_columns)
#         cycle += 1
#     if cycle <= 20:
#         print(f'第{index+1}条合成数据已生成！ cycle={cycle}')
#         synthetic_df = pd.concat([synthetic_df, synthetic_data], ignore_index=True)
#     else:
#         print(f'第{index+1}条输入检索次数过多！')

# 解码
synthetic_data.columns = col_name
for column in columns:
    synthetic_data[column] = label_encoders[column].inverse_transform(synthetic_data[column].astype(int))
synthetic_data.to_csv("Data/german/syn/tabula_llama_e10_b4_t7.csv", index=False)