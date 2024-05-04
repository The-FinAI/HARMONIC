from be_great import GReaT
import pandas as pd
import torch

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

cat_columns = [
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

data = pd.read_csv('/share/fengduanyu/SynData/Data/german/raw/german_train_val.csv')

# Base-Models/llama-2-7b-chat-T
# Base-Models/ditilgpt2

model = GReaT(llm="Base-Models/distilgpt2",
              experiment_dir="results/FT-CP/German-GReaT-GPT2_e30_b1",
              batch_size=1, epochs=30, efficient_finetuning="lora")
model.fit(data)
synthetic_data = model.sample(n_samples=len(data), max_length=1024)
synthetic_data = filter_rows(synthetic_data, data, cat_columns)
#
# while len(synthetic_data) < len(data):
#     new_data = len(data)-len(synthetic_data)
#     print(f"还需生成{new_data}条样本")
#     new_syn_data = model.sample(n_samples=new_data, max_length=1024)
#     new_syn_data = filter_rows(new_syn_data, data, cat_columns)
#     synthetic_data = pd.concat([synthetic_data, new_syn_data], ignore_index=True)
save_path = 'Data/german/syn/great_gpt2_e30_b1.csv'
synthetic_data.to_csv(save_path, index=False)

syn_data = pd.read_csv(save_path)
fil_data = filter_rows(syn_data, data, cat_columns)
print(len(fil_data))

