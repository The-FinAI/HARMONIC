import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data_name, type, label, had_test=False):
    # 读取CSV文件
    data = pd.read_csv('Data/'+data_name+'/raw/'+data_name+'.csv')
    if not had_test:
        if type == 'classification':
            # 根据标签列进行分层抽样，以确保分布一致
            train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[label])
            train_data, val_data = train_test_split(train_val_data, test_size=0.125, random_state=42,
                                                    stratify=train_val_data[label])
        else:
            # 分割数据集
            train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            train_data, val_data = train_test_split(train_val_data, test_size=0.125, random_state=42)

        # 保存分割后的数据集为新的CSV文件
        train_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_train.csv', index=False)
        val_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_val.csv', index=False)
        test_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_test.csv', index=False)
        # train_val_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_train_val.csv', index=False)

    else:
        if type == 'classification':
            # 根据标签列进行分层抽样，以确保分布一致
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[label])
        else:
            # 分割数据集
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        # 保存分割后的数据集为新的CSV文件
        train_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_train.csv', index=False)
        val_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_val.csv', index=False)
        # data.to_csv('Data/' + data_name + '/raw/' + data_name + '_train_val.csv', index=False)


split_data('diabetes', 'classification', 'Outcome')