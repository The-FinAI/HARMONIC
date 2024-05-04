import os.path
import random
import pandas as pd
import json


def get_german_info():
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


def process_prompt(data, mean_list, dicts):
    data_tmp = []
    prompt = 'Evaluate the creditworthiness of a customer with the following financial profile. ' \
             'Respond with only either \'good\' or \'bad\'. For instance, \'The client has a stable ' \
             'income, no previous debts, and owns a property.\' should be classified as \'good\'. \nText: '
    for j in range(len(data)):
        text = ''
        for i in range(len(data[0]) - 1):
            if str(i) not in list(dicts.keys()):
                text = text + 'The state of ' + mean_list[i] + ' is ' + str(data[j][i]) + '. '
            else:
                text = text + 'The state of ' + mean_list[i] + ' is ' + dicts[str(i)][data[j][i]] + '. '
        answer = 'good' if data[j][-1] == 1 else 'bad'
        gold = 0 if data[j][-1] == 1 else 1
        data_tmp.append(
            {'id': j, "query": f"{prompt}'{text}'" + '\nAnswer:', 'answer': answer, "choices": ["good", "bad"],
             "gold": gold, 'text': text})
    return data_tmp


def get_prompt_data(input_path, prompt_path=None, save_flag=False):
    feature_size = 20 + 1  # Target_index = -1
    mean_list, dicts = get_german_info()
    data = pd.read_csv(input_path, sep=',', header=0, names=[i for i in range(feature_size)]).values.tolist()
    data_tmp = process_prompt(data, mean_list, dicts)
    if save_flag:
        save_json(data_tmp, prompt_path)
    return data_tmp


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


def save_ft_data(orig_data, ft_path, dataset_name, split_dev=True, split_train=False, dev_rate=0.1):
    f_write = open(ft_path, "w")
    num_id = 1
    for line in orig_data:
        conversations = [
            {"from": "human", "value": line['instruction'] + line['input']},
            {"from": "assistant", "value": line['output']}
        ]
        # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
        uniq_id = line['id'] if "id" in line else dataset_name + "-" + str(num_id)
        item = {"id": uniq_id, "conversations": conversations}
        f_write.write(json.dumps(item, ensure_ascii=False) + "\n")
        num_id += 1
    f_write.close()
    with open(ft_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [json.loads(line) for line in lines]
    if split_dev:
        ft_dev_path = ft_path.replace('.json', '_dev.json')
        dev_num = int(len(data) * dev_rate)
        dev_data = data[:dev_num]
        save_json(dev_data, ft_dev_path)
    if split_train:
        ft_train_path = ft_path.replace('.json', '_train.json')
        train_data = data[-700:]
        save_json(train_data, ft_train_path)


def save_json(data_list, out_path):
    with open(out_path, 'w') as f_out:
        for item_dict in data_list:
            f_out.write(json.dumps(item_dict, ensure_ascii=False) + '\n')


def llmeval_data(data_path, save_path):
    for i in range(len(data_path)):
        if not os.path.exists(save_path[i]):
            os.makedirs(save_path[i])
        _ = get_prompt_data(input_path=data_path[i], prompt_path=f"{save_path[i]}/test.json", save_flag=True)


def ft_data(methods):
    ft_files = "Data/german/ft/"
    if not os.path.exists(ft_files):
        os.makedirs(ft_files)
    for method in methods:
        rawdata_file = "Data/german/syn/" + method + ".csv"
        ft_data_path = ft_files + method + '_ft.json'
        prompt_data = get_prompt_data(input_path=rawdata_file)  # The prompt_data file will not be saved by default
        instru_data = get_instruction_data(prompt_data=prompt_data)
        save_ft_data(orig_data=instru_data, ft_path=ft_data_path, dataset_name="German")


def main():
    rawdata_file = ["Data/german/raw/german_train_val.csv", "Data/german/raw/german_test.csv", ]
    eval_prompt_path = ["Data/german/llmeval/train", "Data/german/llmeval/test", ]
    llmeval_data(data_path=rawdata_file, save_path=eval_prompt_path)  # Construct a data set for evaluating LLMs

    syn_methods = ['om_e3_b10_t0.7_dict']
    ft_data(methods=syn_methods)


if __name__ == "__main__":
    main()
