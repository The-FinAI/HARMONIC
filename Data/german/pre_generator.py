import argparse

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from pre_llmeval_ft import save_ft_data, save_json


def german_columns():
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

    num_columns = [
        "Duration in month",
        "Credit amount",
        "Installment rate in percentage of disposable income",
        "Present residence since",
        "Age in years",
        "Number of existing credits at this bank",
        "Number of people being liable to provide maintenance for"
    ]
    all_cols = [
        "CheckingStatus", "Duration", "CreditHist", "Purpose", "CreditAmt",
        "Savings", "EmploySince", "InstallRate", "PersStatus", "Debtors",
        "ResidSince", "Property", "Age", "InstallPlans", "Housing",
        "BankCredits", "Job", "MaintLiable", "Telephone", "ForeignWorker", "Status"
    ]
    return cat_columns, num_columns, all_cols


def knn_german(data_path, n_neighbors=6, save_flag=False):
    # 读取原始CSV文件
    cat_columns, num_columns, _ = german_columns()
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
    knn = NearestNeighbors(n_neighbors=n_neighbors)  # 5 neighbors plus itself
    knn.fit(original_data)
    distances, indices = knn.kneighbors(original_data)

    # 构建新的表格
    new_data = pd.DataFrame()
    for i, row in copy_data.iterrows():
        nearest_neighbors_indices = indices[i, 1:]  # Exclude itself
        nearest_neighbors = copy_data.iloc[nearest_neighbors_indices]
        new_data = pd.concat([new_data, nearest_neighbors], ignore_index=True)
        new_data = pd.concat([new_data, copy_data.iloc[[i]]], ignore_index=True)

    # 将结果保存为新的CSV文件
    if save_flag:
        new_data.to_csv('Data/german/generator/knn5.csv', index=False)
    return new_data


def process(df, col_list, mode, sample_format="None", k=6):
    # mode: optional[str, ["train", "syn"]]
    data = df.values.tolist()
    if mode == "syn":
        k -= 1
    numbers_dict = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

    data_tmp = []
    prompt = ""
    if sample_format == "None":
        prompt = "Please generate an approximate data sample based on the following 5 data sample examples."
    elif sample_format == "dict":
        prompt = ("Here are 5 tabular data about user credit scores, "
                  "each containing 20 columns of features and 1 column of labels, "
                  "where the 'Status' column is a binary classification label. "
                  "I will transmit the data to you in JSON format. "
                  "Please generate an approximate sample based on these 5 examples.")
    id = 0
    for j in range(0, len(data), k):
        query = f"{prompt}\n"
        answer = ""
        qa = data[j:j + k]
        for index1, items in enumerate(qa):
            # sample = ""
            # for index2, item in enumerate(items):
            #     sample = sample + f"{col_list[index2]} {str(item)}, " if index2 != len(
            #         items) - 1 else sample + f"{col_list[index2]} {str(item)}."
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
        id += 1
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



def filter_label(target_output, outputs, r):
    if target_output.iloc[-1] == 1:
        if outputs.sum().iloc[-1] <= r:
            # 如果标签为1，则过滤大小为k且不超过r条标签为1的样本
            return False
        else:
            return True
    elif target_output.iloc[-1] == 0:
        if outputs.sum().iloc[-1] >= 5 - r:
            return False
        else:
            return True

# 检查组是否错误
def filter_data(df):
    cases = []
    filtered_data = pd.DataFrame(columns=df.columns)
    for i in range(0, len(df), 6):
        group = df.iloc[i:i + 6]
        outputs = group.iloc[:5]
        target_output = group.iloc[-1]

        if filter_label(target_output, outputs, r=2):
            filtered_data = pd.concat([filtered_data, group])
        else:
            cases.append(i + 7)
    print("过滤后数据大小:", len(df) // 6 - len(cases))
    return filtered_data


def random_sampling(original_path, random_seed=416):
    df = pd.read_csv(original_path)
    sampled_df = pd.DataFrame(columns=df.columns)
    for _ in range(800):
        seed_state = _ + random_seed
        sampled_rows = df.sample(n=5, random_state=seed_state)
        sampled_df = pd.concat([sampled_df, sampled_rows], ignore_index=True)
    return sampled_df


def ft_generator_data(raw_data_path, ft_path, cols, is_fil):
    knn_data = knn_german(raw_data_path)
    if is_fil:
        knn_data = filter_data(knn_data)
    ins_data = process(df=knn_data, col_list=cols, mode="train", sample_format="dict")
    save_ft_data(orig_data=ins_data, ft_path=ft_path, dataset_name="German", split_dev=True)


def pre_generator_data(real_data_path, pre_path, cols):
    sample_df = random_sampling(real_data_path, random_seed=416)
    ins_data = process(df=sample_df, col_list=cols, mode="syn", sample_format="dict")
    save_ft_data(orig_data=ins_data, ft_path=pre_path, dataset_name="German", split_dev=False)


def main(args):
    ft_data_path = "Data/german/generator/knn5_dict_ft.json"
    pre_path = "Data/german/generator/pre_knn5_dict.json"
    _, _, columns_name = german_columns()
    ft_generator_data(raw_data_path=args.data_path, ft_path=ft_data_path, cols=columns_name, is_fil=False)
    pre_generator_data(real_data_path=args.data_path, pre_path=pre_path, cols=columns_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, default="Data/german/raw/german_train.csv",
                        hele="", required=True)
    parser.add_argument("seed", type=int, default=416, help="")
    parser.add_argument('knn_n', type=int, default=5)
    arguments = parser.parse_args()

    main(arguments)
