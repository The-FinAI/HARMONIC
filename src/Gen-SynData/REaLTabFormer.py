# pip install realtabformer
import pandas as pd
from realtabformer import REaLTabFormer
import numpy as np
import random

# # 设置随机种子
# seed_value = 4
# np.random.seed(seed_value)
# random.seed(seed_value)
data_name='adult'
df = pd.read_csv(f'Data/{data_name}/raw/{data_name}_train.csv')
# df = pd.read_csv('/media/data1/jiangsj/Fraud-4-57/row_data/train_set.csv')

# NOTE: Remove any unique identifiers in the
# data that you don't want to be modeled.
seed_value=1026  #1029(0), 1028(1), 1027(2), 1026(3), 1025(4)
# Non-relational or parent table.
rtf_model = REaLTabFormer(
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=100,
    random_state=seed_value)

# gradient_accumulation_steps=4,
# logging_steps=100

# Fit the model on the dataset.
# Additional parameters can be
# passed to the `.fit` method.
rtf_model.fit(df)

# Save the model to the current directory.
# A new directory `rtf_model/` will be created.
# In it, a directory with the model's
# experiment id `idXXXX` will also be created
# where the artefacts of the model will be stored.
rtf_model.save("rtf_model/")

# # 设置随机种子确保每次生成的合成数据一致
# np.random.seed(seed_value)
# random.seed(seed_value)
# Generate synthetic data with the same
# number of observations as the real dataset.
samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f'Data/{data_name}/syn/{data_name}_rtf{1029-seed_value}.csv', index=False)

# Load the saved model. The directory to the
# experiment must be provided.
# rtf_model2 = REaLTabFormer.load_from_dir(
#     path="rtf_model/idXXXX")