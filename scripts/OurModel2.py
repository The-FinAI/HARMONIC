import argparse
import pandas as pd
import json
import random
import numpy as np
import subprocess
import sys

sys.path.append('src/Gen-SynData/ourmodel')

from ourmodel import OurModel

def load_data(data_name):
    processed_data = []
    with open(f'Data/{data_name}/generator/knn_syn.json', 'r') as file:
        for line in file:
            temp = json.loads(line)
            processed_data.append(temp["conversations"][0]["value"])
    return processed_data

def get_columns(df):
    features = df.columns.tolist()
    num_f = df.iloc[:, :-1].select_dtypes(include=['number']).columns.tolist()
    cat_f = df.iloc[:, :-1].select_dtypes(exclude=['number']).columns.tolist()

    return cat_f, num_f, features

def sample(data_name, model, real_data, processed_data, sample_num, temperature, max_length, seed):
    cat_columns, num_columns, columns= get_columns(real_data)
    synthetic_df = pd.DataFrame(columns=columns)
    # todo 随机种子
    np.random.seed(seed)
    random.seed(seed)
    for index in range(sample_num):
        # todo 随机种子
        synthetic_data = model.tabula_sample(starting_prompts=processed_data[index],
                                             temperature=temperature, max_length=max_length)
        synthetic_df = pd.concat([synthetic_df, synthetic_data], ignore_index=True)
    synthetic_df.to_csv(f'Data/{data_name}/syn/ourmodel.csv', index=False)
    return

def main(args):
    data_name = args.data_name
    sample_num = args.sample_num
    temperature = args.temperature
    max_length = args.max_length
    seed = args.seed
    real_data = pd.read_csv(f'Data/{data_name}/raw/{data_name}_train.csv')
    pre_data = load_data(data_name)
    model = OurModel(llm=f'results/FT-LLMs/llama2-7b-chat-gen/${data_name}-gen',
                     data=real_data)
    sample(data_name=data_name, model=model, real_data=real_data, processed_data=pre_data,
           sample_num=sample_num, temperature=temperature, max_length=max_length, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_name", type=str, default="diabetes")
    parser.add_argument("sample_num", type=int, help="The sampling count")
    parser.add_argument("seed", type=int, default=2416,
                        help="Constructing the random seed for generating synthetic data")
    parser.add_argument("temperature", type=float, default=0.7)
    parser.add_argument("max_length", type=int, default=2048)
    parser.add_argument("task_type", type=str,
                        choices=['binary classification', 'multi classification', 'regression'])
    args = parser.parse_args()

    # 生成合成数据
    main(args)

    # 生成微调下游任务需要的数据
    subprocess.run(['python3.9', "scripts/pre_llmeval_ft.py",
                    f'{args.data_name}', "ourmodel", f'{args.task_type}'], check=True)
