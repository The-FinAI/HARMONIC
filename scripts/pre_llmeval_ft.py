import os.path
import random
import pandas as pd
import json
import argparse


def german_info():
    mean_list = ['Status of existing checking account', 'Duration in month', 'Credit history', 'Purpose',
                 'Credit amount', 'Savings account or bonds', 'Present employment since',
                 'Installment rate in percentage of disposable income', 'Personal status and sex',
                 ' Other debtors or guarantors', 'Present residence since', 'Property', 'Age in years',
                 'Other installment plans', 'Housing', 'Number of existing credits at this bank', 'Job',
                 'Number of people being liable to provide maintenance for', 'Telephone', 'foreign worker'
                 ]

    dicts = {'0': {'A11': 'smaller than 0 DM', 'A12': 'bigger than 0 DM but smaller than 200 DM',
                   'A13': 'bigger than 200 DM OR salary assignments for at least 1 year',
                   'A14': 'no checking account'},
             '2': {'A30': 'no credits taken or all credits paid back duly',
                   'A31': 'all credits at this bank paid back duly',
                   'A32': 'existing credits paid back duly till now',
                   'A33': 'delay in paying off in the past',
                   'A34': 'critical account or other credits existing (not at this bank)'},
             '3': {'A40': 'car (new)',
                   'A41': 'car (used)',
                   'A42': 'furniture or equipment',
                   'A43': 'radio or television',
                   'A44': 'domestic appliances',
                   'A45': 'repairs',
                   'A46': 'education',
                   'A47': 'vacation',
                   'A48': 'retraining',
                   'A49': 'business',
                   'A410': 'others'},
             '5': {'A61': 'smaller than 100 DM',
                   'A62': 'bigger than 100 smaller than  500 DM',
                   'A63': 'bigger than 500 smaller than 1000 DM',
                   'A64': 'bigger than 1000 DM',
                   'A65': 'unknown or no savings account'},
             '6': {'A71': 'unemployed',
                   'A72': 'smaller than 1 year',
                   'A73': 'bigger than 1  smaller than 4 years',
                   'A74': 'bigger than 4  smaller than 7 years',
                   'A75': 'bigger than 7 years'},
             '8': {'A91': 'male: divorced or separated',
                   'A92': 'female: divorced or separated or married',
                   'A93': 'male and single',
                   'A94': 'male and married or widowed',
                   'A95': 'female and single'},
             '9': {'A101': 'none',
                   'A102': 'co-applicant',
                   'A103': 'guarantor'},
             '11': {'A121': 'real estate',
                    'A122': 'building society savings agreement or life insurance',
                    'A123': 'car or other',
                    'A124': 'unknown or no property'},
             '13': {'A141': 'bank',
                    'A142': 'stores',
                    'A143': 'none'},
             '14': {'A151': 'rent',
                    'A152': 'own',
                    'A153': 'for free'},
             '16': {'A171': 'unemployed or unskilled or non-resident',
                    'A172': 'unskilled or resident',
                    'A173': 'skilled employee or official',
                    'A174': 'management or self-employed or highly qualified employee or officer'},
             '18': {'A191': 'none',
                    'A192': 'yes, registered under the customers name'},
             '19': {'A201': 'yes',
                    'A202': 'no'},
             }
    return mean_list, dicts


# def process_prompt(data, mean_list, dicts):
#     data_tmp = []
#     prompt = 'Evaluate the creditworthiness of a customer with the following financial profile. ' \
#              'Respond with only either \'good\' or \'bad\'. For instance, \'The client has a stable ' \
#              'income, no previous debts, and owns a property.\' should be classified as \'good\'. \nText: '
#     for j in range(len(data)):
#         text = ''
#         for i in range(len(data[0]) - 1):
#             if str(i) not in list(dicts.keys()):
#                 text = text + 'The state of ' + mean_list[i] + ' is ' + str(data[j][i]) + '. '
#             else:
#                 text = text + 'The state of ' + mean_list[i] + ' is ' + dicts[str(i)][data[j][i]] + '. '
#         answer = 'good' if data[j][-1] == 1 else 'bad'
#         gold = 0 if data[j][-1] == 1 else 1
#         data_tmp.append(
#             {'id': j, "query": f"{prompt}'{text}'" + '\nAnswer:', 'answer': answer, "choices": ["good", "bad"],
#              "gold": gold, 'text': text})
#     return data_tmp

def get_eval_data(input_path, task, task_type):
    df = pd.read_csv(input_path, sep=',', header=0)
    data = df.values.tolist()
    cols = df.columns.tolist()
    data_tmp = []
    prompt = prompt_info[task]
    for j in range(len(data)):
        text = ''
        for i in range(len(data[0]) - 1):
            if task == "german":
                mean_list, dicts = german_info()
                if str(i) not in list(dicts.keys()):
                    text = text + 'The state of ' + mean_list[i] + ' is ' + str(data[j][i]) + ', '
                else:
                    text = text + 'The state of ' + mean_list[i] + ' is ' + dicts[str(i)][data[j][i]] + ', '
            else:
                text = text + 'The state of ' + cols[i] + ' is ' + str(data[j][i]) + ', '
        text = text[:-2] + "."
        answer = None
        if task_type == "binary classification" or task_type == "multi classification":
            choices, gold = None, None
            if task == "german":
                choices = ["bad", "good"]
                gold = int(data[j][-1])
            elif task == "diabetes":
                choices = ["negative", "positive"]
                gold = int(data[j][-1])
            elif task == "adult":
                choices = ["good", "bad"]
                gold = 0 if str(data[j][-1]) == ">=50k" else 1
            elif task == "buddy":
                choices = ["A", "B", "C"]
                gold = int(data[j][-1])
            answer = choices[gold]
            data_tmp.append(
                {'id': j, "query": f"{prompt}'{text}'" + '\nAnswer: ', 'answer': answer, "choices": choices,
                 "gold": gold, 'text': text})
        elif task_type == "regression":
            answer = int(data[j][-1])
            data_tmp.append(
                {'id': j, "query": f"{prompt}'{text}'" + '\nAnswer: ', 'answer': answer, 'text': text})
    return data_tmp


# def get_prompt_data(data_name, input_path):
#     # tep_data = pd.read_csv(input_path)
#     # feature_size = 20 + 1  # Target_index = -1
#     # feature_size = tep_data.shape[1]
#     # data = pd.read_csv(input_path, sep=',', header=0, names=[i for i in range(feature_size)]).values.tolist()
#     # if data_name == 'german':
#     #     mean_list, dicts = get_german_info()
#     data_tmp = get_eval_data(input_path=data_name, task=data_name, task_type=args.task_type)
#     return data_tmp


def get_instruction_data(prompt_data):
    output_data = []
    for line in prompt_data:
        data = line
        query = data.get('query', '')
        answer = data.get('answer', '')

        new_data = {
            "instruction": query,
            "input": "",
            "output": answer
        }
        output_data.append(new_data)
    return output_data


# def save_ft_data(orig_data, ft_path, dataset_name, split_dev=True, split_train=False, dev_rate=0.1):
#     f_write = open(ft_path, "w")
#     num_id = 1
#     for line in orig_data:
#         conversations = [
#             {"from": "human", "value": line['instruction'] + line['input']},
#             {"from": "assistant", "value": line['output']}
#         ]
#         # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
#         uniq_id = line['id'] if "id" in line else dataset_name + "-" + str(num_id)
#         item = {"id": uniq_id, "conversations": conversations}
#         f_write.write(json.dumps(item, ensure_ascii=False) + "\n")
#         num_id += 1
#     f_write.close()
#     with open(ft_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#     data = [json.loads(line) for line in lines]
#     if split_dev:
#         ft_dev_path = ft_path.replace('.json', '_dev.json')
#         dev_num = int(len(data) * dev_rate)
#         dev_data = data[:dev_num]
#         save_json(dev_data, ft_dev_path)
#     if split_train:
#         ft_train_path = ft_path.replace('.json', '_train.json')
#         train_data = data[-700:]
#         save_json(train_data, ft_train_path)

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


# def save_json(data_list, out_path):
#     with open(out_path, 'w') as f_out:
#         for item_dict in data_list:
#             f_out.write(json.dumps(item_dict, ensure_ascii=False) + '\n')


# def llmeval_data(data_path, save_path):
#     for i in range(len(data_path)):
#         if not os.path.exists(save_path[i]):
#             os.makedirs(save_path[i])
#         _ = get_prompt_data(input_path=data_path[i], prompt_path=f"{save_path[i]}/test.json", save_flag=True)



# def ft_data(data_name, method, task_type):
#     ft_files = f'Data/{data_name}/ft/'
#     if not os.path.exists(ft_files):
#         os.makedirs(ft_files)
#     # rawdata_file = f'Data/{data_name}/syn/{method}.csv'
#     pre_syn_path = f'Data/{data_name}/{method}.csv'
#     ft_data_path = f'{ft_files}{method}_ft.json'
#     # prompt_data = get_prompt_data(data_name=data_name, input_path=rawdata_file)
#     prompt_data = get_eval_data(input_path=pre_syn_path, task=data_name, task_type=task_type)
#     instru_data = get_instruction_data(prompt_data=prompt_data)
#     save_ft_data(orig_data=instru_data, ft_path=ft_data_path)



def main(args):
    data_name = args.data_name
    syn_method = args.method_name
    task_type = args.task_type
    # ft_data(data_name=args.data_name, method=args.method_name, task_type=args.task_type)
    ft_files = f'Data/{data_name}/ft/'
    if not os.path.exists(ft_files):
        os.makedirs(ft_files)
    pre_syn_path = f'Data/{data_name}/{syn_method}.csv'
    ft_data_path = f'{ft_files}{syn_method}_ft.json'
    prompt_data = get_eval_data(input_path=pre_syn_path, task=data_name, task_type=task_type)
    instru_data = get_instruction_data(prompt_data=prompt_data)
    save_ft_data(orig_data=instru_data, ft_path=ft_data_path)

## 已修改
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_name", type=str, default="diabetes")
    parser.add_argument("method_name", type=str, default='great')
    parser.add_argument("task_type", type=str, choices=['binary classification', 'multi classification', 'regression'])
    args = parser.parse_args()

    prompt_info = {
        "german": "Evaluate the creditworthiness of a customer with the following financial profile. "
                  "Respond with only either 'good' or 'bad'. \nText: ",
        "diabetes": "Predict whether a patient has diabetes, based on certain diagnostic measurements. "
                  "Respond with only either 'negative' or 'positive'. \nText: ",
        "adult": "Determine whether a person makes over $50K a year based on personal attributes. "
                 "Respond with only either 'good' or 'bad'. \nText: ",
        "buddy": "Detect the breed of an animal based on its condition, appearance, and other factors. "
                 "Respond with only either 'A', 'B' or 'C'. \nText: ",
        "abalone": "Predict the age of abalone from physical measurements. Respond with an integer. \nText: ",
        "california": "Predict the median house price in California based on features such as population count, "
                      "median income, median house age, etc. Respond with an integer. \nText: ",
    }


    main(args)
