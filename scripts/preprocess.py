import argparse
import os.path
import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors


def save_json(data_list, out_path):
    with open(out_path, 'w') as f_out:
        for item_dict in data_list:
            f_out.write(json.dumps(item_dict, ensure_ascii=False) + '\n')


def save_ft_data(orig_data, ft_path):
    f_write = open(ft_path, "w")
    num_id = 1
    for line in orig_data:
        conversations = [
            {"from": "human", "value": line['instruction'] + line['input']},
            {"from": "assistant", "value": line['output']}
        ]
        # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
        uniq_id = line['id'] if "id" in line else str(num_id)
        item = {"id": uniq_id, "conversations": conversations}
        f_write.write(json.dumps(item, ensure_ascii=False) + "\n")
        num_id += 1
    f_write.close()


def get_columns(path):
    df = pd.read_csv(path)
    features = df.columns.tolist()

    num_f = df.iloc[:, :-1].select_dtypes(include=['number']).columns.tolist()
    cat_f = df.iloc[:, :-1].select_dtypes(exclude=['number']).columns.tolist()

    return cat_f, num_f, features


def knn_fit(data_path, n, features):
    # 读取原始CSV文件
    cat_columns, num_columns = features[0], features[1]
    original_data = pd.read_csv(data_path)
    copy_data = original_data.copy()
    # 将分类型特征转化为数值型特征
    label_encoder = LabelEncoder()
    for column in cat_columns:
        original_data[column] = label_encoder.fit_transform(original_data[column])

    # 归一化数值型特征
    # numeric_columns = original_data.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    original_data[num_columns] = scaler.fit_transform(original_data[num_columns])

    # 使用KNN算法找到最近的5条数据
    knn = NearestNeighbors(n_neighbors=n+1)  # 5 neighbors plus itself
    knn.fit(original_data)
    distances, indices = knn.kneighbors(original_data)

    # 构建新的表格
    new_data = pd.DataFrame()
    for i, row in copy_data.iterrows():
        nearest_neighbors_indices = indices[i, 1:]  # Exclude itself
        nearest_neighbors = copy_data.iloc[nearest_neighbors_indices]
        new_data = pd.concat([new_data, nearest_neighbors], ignore_index=True)
        new_data = pd.concat([new_data, copy_data.iloc[[i]]], ignore_index=True)
    return new_data


def process(df, mode, k, sample_format):
    """
    param:
    """
    # mode: optional[str, ["train", "syn"]]
    data = df.values.tolist()
    col_list = df.columns.to_list()
    if mode == "syn":
        k -= 1
    numbers_dict = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

    data_tmp = []
    # prompt = ""
    if sample_format == "None":
        prompt = " "  # todo
    elif sample_format == "dict":
        prompt = f"Here are {k-1} tabular data about {arguments.des}, " \
             f"each containing {len(col_list)} columns of features and 1 column of labels, " \
             f"where the {col_list[-1]} column is a {arguments.task_type} label. " \
             f"I will transmit the data to you in JSON format. " \
             f"Please generate an approximate sample based on these {k-1} examples."

    for j in range(0, len(data), k):
        query = f"{prompt}\n"
        answer = ""
        qa = data[j:j + k]
        for index1, items in enumerate(qa):
            sample = get_line_sample(col_list=col_list, line_list=items, sample_format=sample_format)
            if mode == "train":
                if index1 != len(qa) - 1:
                    query = query + f"Example {numbers_dict[index1 + 1]}: {sample}\n"
                if index1 == len(qa) - 1:
                    query = query + "Generate one sample: "
                    answer = sample
            elif mode == "syn":
                query = query + f"Example {numbers_dict[index1 + 1]}: {sample}\n"
                if index1 == len(qa) - 1:
                    query = query + "Generate one sample: "
                    answer = ""

        data_tmp.append({
            "instruction": query,
            "input": "",
            "output": answer,
        })
    return data_tmp


def get_line_sample(col_list, line_list, sample_format):
    sample = ""
    for index2, item in enumerate(line_list):
        if sample_format == "dict":
            sample = sample + f'"{col_list[index2]}": "{str(item)}", '
        elif sample_format == "None":
            sample + f"{col_list[index2]} {str(item)}, " if index2 != len(
                line_list) - 1 else sample + f"{col_list[index2]} {str(item)}."
    if sample_format == "dict":
        sample = "{" + sample[:-2] + "}"
    return sample


def filter_label(target_output, outputs, r, n):
    if target_output.iloc[-1] == 1:
        if outputs.sum().iloc[-1] <= r:
            # 如果标签为1，则过滤大小为k且不超过r条标签为1的样本
            return False
        else:
            return True
    elif target_output.iloc[-1] == 0:
        if outputs.sum().iloc[-1] >= n - r:
            return False
        else:
            return True


# 检查组是否错误
def filter_data(df, r, n):
    """"
    r:
    """
    cases = []
    filtered_data = pd.DataFrame(columns=df.columns)
    for i in range(0, len(df), n+1):
        group = df.iloc[i:i + n+1]
        outputs = group.iloc[:n]
        target_output = group.iloc[-1]

        if filter_label(target_output, outputs, r, n):
            filtered_data = pd.concat([filtered_data, group])
        else:
            cases.append(i + n + 2)
    print("过滤后数据大小:", len(df) // (n+1) - len(cases))
    return filtered_data


def random_sampling(original_path, random_seed, n, sample_num):
    df = pd.read_csv(original_path)
    sampled_df = pd.DataFrame(columns=df.columns)
    for _ in range(sample_num):
        seed_state = _ + random_seed
        sampled_rows = df.sample(n=n, random_state=seed_state)
        sampled_df = pd.concat([sampled_df, sampled_rows], ignore_index=True)
    return sampled_df


def ft_generator_data(raw_data_path, ft_path, cols, is_fil, n, sample_format):
    """
    raw_data_path:
    fy_path:
    cols: [cat, num, all]
    is_fill:
    seed:
    n:
    """
    # data_path, n, features
    knn_data = knn_fit(data_path=raw_data_path, n=n, features=cols)
    if is_fil:
        knn_data = filter_data(df=knn_data, r=2, n=n)
    # process(df, col_list, mode, k)
    ins_data = process(df=knn_data, k=n+1, mode="train", sample_format=sample_format)
    save_ft_data(orig_data=ins_data, ft_path=ft_path)


def pre_generator_data(real_data_path, pre_path, seed, n, sample_format, sample_num):
    sample_df = random_sampling(real_data_path, random_seed=seed, n=n, sample_num=sample_num)
    ins_data = process(df=sample_df, k=n+1, mode="syn", sample_format=sample_format)
    save_ft_data(orig_data=ins_data, ft_path=pre_path)


def main(args):
    data_path = f"Data/{args.data_name}/raw/{args.data_name}_train.csv"
    generator_path = f"Data/{args.data_name}/generator"
    if not os.path.exists(generator_path):
        os.makedirs(generator_path)

    features = get_columns(data_path)  # list[list[str]]: (cat, num, all)
    ft_generator_data(
        raw_data_path=data_path,
        ft_path=f"{generator_path}/knn_train.json",
        cols=features,
        is_fil=True,
        n=args.knn_n,
        sample_format=args.re_format
    )
    pre_generator_data(
        real_data_path=data_path,
        pre_path=f"{generator_path}/knn_syn.json",
        n=args.knn_n,
        seed=args.seed,
        sample_format=args.re_format,
        sample_num=args.sample_num,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_name", type=str, default="diabetes")
    # parser.add_argument("prompt", type=str, default="")
    parser.add_argument("seed", type=int, default=416, help="")
    parser.add_argument('knn_n', type=int, default=5)
    parser.add_argument("task_type", type=str)
    parser.add_argument("des", type=str, help="A simple description for this data")
    parser.add_argument("re_format", type=str, help=" ")
    parser.add_argument("sample_num", type=int, help="")

    arguments = parser.parse_args()
    main(arguments)

    """
    "diabetes": ["binary classification", "diabetic patients"]
    "german": ["binary classification", "user credit scores"]
    """
