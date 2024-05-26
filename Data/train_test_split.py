import pandas as pd
from sklearn.model_selection import train_test_split

# def split_data(data_name, type, label, had_test=False):
#     # 读取CSV文件
#     data = pd.read_csv('Data/'+data_name+'/raw/'+data_name+'.csv')
#     if not had_test:
#         if type == 'classification':
#             # 根据标签列进行分层抽样，以确保分布一致
#             train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[label])
#             train_data, val_data = train_test_split(train_val_data, test_size=0.125, random_state=42,
#                                                     stratify=train_val_data[label])
#         else:
#             # 分割数据集
#             train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#             train_data, val_data = train_test_split(train_val_data, test_size=0.125, random_state=42)
#
#         # 保存分割后的数据集为新的CSV文件
#         train_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_train.csv', index=False)
#         val_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_val.csv', index=False)
#         test_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_test.csv', index=False)
#         # train_val_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_train_val.csv', index=False)
#
#     else:
#         if type == 'classification':
#             # 根据标签列进行分层抽样，以确保分布一致
#             train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[label])
#         else:
#             # 分割数据集
#             train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
#         # 保存分割后的数据集为新的CSV文件
#         train_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_train.csv', index=False)
#         val_data.to_csv('Data/' + data_name + '/raw/' + data_name + '_val.csv', index=False)
#         # data.to_csv('Data/' + data_name + '/raw/' + data_name + '_train_val.csv', index=False)
#
#
# split_data('adult', 'classification', 'class')

# def split_data(data_name, type, label=None):
#
#     for seed in range(3):
#         df = pd.read_csv(f'Data/{data_name}/syn/{data_name}_smote{seed}.csv')
#         if type == 'classification':
#             # 分层采样，按照分类列的名称进行分层
#             _, sampled_df = train_test_split(df, test_size=5000, stratify=df[label], random_state=42)
#         else:
#             # # 计算分位数
#             # num_quantiles = 10  # 可以根据需要调整分位数的数量
#             # df['quantile'] = pd.qcut(df[label], num_quantiles, labels=False)
#             #
#             # # 分层采样
#             # sampled_df = df.groupby('quantile').apply(lambda x: x.sample(frac=5000 / 13000)).reset_index(drop=True)
#             #
#             # # 保存采样结果
#             # sampled_df.drop(columns='quantile', inplace=True)  # 删除辅助的分位数列
#             _, sampled_df = train_test_split(df, test_size=5000, random_state=42)
#
#         # # 检查采样后的分类分布
#         # print(sampled_df['category'].value_counts(normalize=True))
#         # print(df['category'].value_counts(normalize=True))
#
#         # 保存采样结果到新的CSV文件
#         sampled_df.to_csv(f'Data/{data_name}/new_syn/{data_name}_smote{seed}.csv', index=False)
#
#
# split_data('california', 'reg')

def split_data(data_name, type, label=None):

    for split in ['train']:
        df = pd.read_csv(f'Data/{data_name}/raw/{data_name}_{split}.csv')
        if type == 'classification':
            # 分层采样，按照分类列的名称进行分层
            _, sampled_df = train_test_split(df, test_size=5000, stratify=df[label], random_state=42)
        else:
            _, sampled_df = train_test_split(df, test_size=5000, random_state=42)

        # 保存采样结果到新的CSV文件
        sampled_df.to_csv(f'Data/{data_name}/raw/{data_name}_{split}5000.csv', index=False)


split_data('buddy', 'classification', 'breed_category')