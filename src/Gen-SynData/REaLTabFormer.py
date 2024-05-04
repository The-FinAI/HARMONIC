# pip install realtabformer
import pandas as pd
from realtabformer import REaLTabFormer

df = pd.read_csv('Data/german/raw/german_train_val.csv')
# df = pd.read_csv('/media/data1/jiangsj/Fraud-4-57/row_data/train_set.csv')

# NOTE: Remove any unique identifiers in the
# data that you don't want to be modeled.

# Non-relational or parent table.
rtf_model = REaLTabFormer(
    model_type="tabular",
    batch_size=4,
    )

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

# Generate synthetic data with the same
# number of observations as the real dataset.
samples = rtf_model.sample(n_samples=len(df))
samples.to_csv('Data/german/syn/realtabformer_llama.csv', index=False)

# Load the saved model. The directory to the
# experiment must be provided.
# rtf_model2 = REaLTabFormer.load_from_dir(
#     path="rtf_model/idXXXX")