from tabula import Tabula
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import random
import json

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

model = Tabula(llm="Base-Models/llama-2-7b-chat-T", experiment_dir="results/FT-CP/German-TabuLa-LLaMA2-Chat_e10_b10",
               batch_size=10, epochs=10, efficient_finetuning="lora")

# model.model.load_state_dict(torch.load("pretrained-model/model.pt"), strict=False)

model.fit(data)

save = torch.save(model.model.state_dict(), "results/FT-LLMs/German-TabuLa-LLaMA2-Chat/model_e10_b10_t7.pt")

synthetic_data = model.sample(n_samples=len(data), temperature=0.7, max_length=1024)

# 解码
synthetic_data.columns = col_name
for column in columns:
    synthetic_data[column] = label_encoders[column].inverse_transform(synthetic_data[column].astype(int))
synthetic_data.to_csv("Data/german/syn/tabula_llama_e10_b10_t7.csv", index=False)