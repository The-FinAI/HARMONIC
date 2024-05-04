import random
import pandas as pd
import json


def process(data, column_names):
    data_tmp = []
    prompt = f"请根据本案件的量刑要素预测其相应的刑期（以月为单位）。该案件包含的量刑要素有："
    for j in range(len(data)):
        query = ""

        for i in range(len(column_names)-1):
            if data[j][i] == 1:
                query = query + f"{column_names[i]}" + "，"
        query_list = list(query)
        query_list.pop()
        query_list.append("。")
        query = "".join(query_list)
        answer = data[j][-1]
        data_tmp.append(
            {'id': j, "query": f"{prompt}'{query}'", 'answer': answer})
    return data_tmp


def get_prompt_data(input_path, prompt_path):
    feature_size = 112 + 1  # Target_index = -1
    data = pd.read_csv(input_path, sep=',', header=0)
    col_names = data.columns.to_list()
    data = data.values.tolist()
    data_tmp = process(data, col_names)
    with open(prompt_path, 'w', encoding="utf-8") as f_out:
        for new_data in data_tmp:
            json.dump(new_data, f_out, ensure_ascii=False)
            f_out.write('\n')
    return data_tmp



if __name__ == "__main__":
    rawdata_file = ["/home/wangyx/relat_to_local/mydata/SynData/raw-data/fraud/jzh_fraud_train_val.csv",
                    "/home/wangyx/relat_to_local/mydata/SynData/raw-data/fraud/jzh_fraud_test.csv",
                    ]
    prompt_path = ["/home/wangyx/relat_to_local/mydata/SynData/Eval-Metrics/eval-llm/data/fraud_train/test.json",
                   "/home/wangyx/relat_to_local/mydata/SynData/Eval-Metrics/eval-llm/data/fraud_test/test.json",]
    for i in range(len(rawdata_file)):
        get_prompt_data(input_path=rawdata_file[i], prompt_path=prompt_path[i])