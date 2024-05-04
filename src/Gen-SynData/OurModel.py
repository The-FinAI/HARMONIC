from ourmodel import OurModel
import pandas as pd
import torch
# from sklearn.preprocessing import LabelEncoder
# import random
import json

# def choose_begining(df, row_index):
#     col_names = df.columns.tolist()
#     random_name = random.choice(col_names)
#     # random_name = 'Status'
#     cor_value = df.loc[row_index, random_name]
#     # starting_prompts = random_name + ' ' + str(cor_value)
#     # label_value = df.loc[row_index, 'Status']
#     # starting_prompts = 'Status' + ' ' + str(label_value) + ',' + ' ' + random_name + ' ' + str(cor_value)
#     starting_prompts = random_name + ' ' + str(cor_value)
#     return starting_prompts

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
    if len(syn_data) == 0:
        return True
    elif syn_data.isna().any().any():
        return True
    elif not all(element.isdigit() for element in syn_data.loc[0, num_columns]):
        return True
    else:
        return False

real_data = pd.read_csv("Data/german/raw/german_train_val.csv")
# 使用labelencoder将分类特征转化为数值
col_name = real_data.columns.tolist()
# label_encoder = LabelEncoder()
# columns = [
#     "Status of existing checking account",
#     "Credit history",
#     "Purpose",
#     "Savings account/bonds",
#     "Present employment since",
#     "Personal status and sex",
#     "Other debtors / guarantors",
#     "Property",
#     "Other installment plans",
#     "Housing",
#     "Job",
#     "Telephone",
#     "foreign worker"
# ]


# Rename the columns based on the suggested abbreviations
real_data.columns = [
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

processed_data = []
# 打开 JSON 文件
with open("Data/german/generator/pre_knn5_dict.json", 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 解析 JSON 数据
        temp = json.loads(line)
        processed_data.append(temp["conversations"][0]["value"])


model = OurModel(llm="/share/fengduanyu/SynData/results/FT-LLMs/German-OurModel-LLaMA2-Chat_e7_b10_dict",
               data = real_data)


synthetic_df = pd.DataFrame(columns=real_data.columns.tolist())
for index in range(800):
    # starting_prompts = choose_begining(data, index)
    synthetic_data = model.tabula_sample(starting_prompts=processed_data[index], temperature=0.7, max_length=2048)
    synthetic_data = filter_rows(synthetic_data, real_data, cat_columns)
    # 判断是否存在某些列生成None值，如果存在，则重新生成
    cycle = 0
    while ana_conditon(synthetic_data) and cycle <= 20:
        synthetic_data = model.tabula_sample(starting_prompts=processed_data[index], temperature=0.7, max_length=2048)
        synthetic_data = filter_rows(synthetic_data, real_data, cat_columns)
        cycle += 1

    # while synthetic_data.isna().any().any() and not all(isinstance(element, (int, float)) for element in synthetic_data[num_columns]):
    #     synthetic_data = model.tabula_sample(starting_prompts=starting_prompts, temperature=1.0, max_length=512)
    if cycle <= 20:
        print(f'第{index+1}条合成数据已生成！ cycle={cycle}')
        synthetic_df = pd.concat([synthetic_df, synthetic_data], ignore_index=True)
    else:
        print(f'第{index+1}条输入检索次数过多！')

# synthetic_data.to_csv("tabular_7000_init.csv", index=False)
# synthetic_data = pd.read_csv('tabular_7000_init.csv')

# # 解码
# synthetic_data.columns = col_name
# for column in columns:
#     synthetic_data[column] = label_encoders[column].inverse_transform(synthetic_data[column].astype(int))
# synthetic_data.to_csv("../gr/tabula/tabular_llama_10_0.3.csv", index=False)

synthetic_df.columns = col_name
# for column in columns:
#     synthetic_df[column] = label_encoders[column].inverse_transform(synthetic_df[column].astype(int))
synthetic_df.to_csv("Data/german/syn/om_e7_b10_t0.7_dict.csv", index=False)