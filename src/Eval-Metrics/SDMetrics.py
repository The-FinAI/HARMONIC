import argparse

import pandas as pd
import numpy as np
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import NewRowSynthesis
from sdmetrics.single_column import BoundaryAdherence
from sdmetrics.single_column import RangeCoverage
from sdmetrics.single_column import CategoryCoverage


def col_shape(df):
    kolmogorov_smirnov = 0
    total_variation_distance = 0
    n = len(df['Metric'])
    count = 0
    for i in range(n):
        if df['Metric'][i] == 'KSComplement':
            kolmogorov_smirnov += df['Score'][i]
            count += 1
        else:
            total_variation_distance += df['Score'][i]

    ks_mean = kolmogorov_smirnov / count if count > 0 else 0
    tvd_mean = total_variation_distance / (n-count) if (n-count) > 0 else 0
    return ks_mean, tvd_mean


def col_pair_trend(df):
    contingency_table_similarity = 0
    person_correlation_similarity = 0
    n = len(df['Metric'])
    count = 0
    for i in range(n):
        if df['Metric'][i] == 'ContingencySimilarity':
            contingency_table_similarity += df['Score'][i]
            count += 1
        else:
            person_correlation_similarity += df['Score'][i]
    cts_mean = contingency_table_similarity / count if count > 0 else 0
    pcs_mean = person_correlation_similarity / (n-count) if (n-count) > 0 else 0

    return cts_mean, pcs_mean

def coverage(df1, df2, dict):
    num_score = []
    cat_score = []
    for key, value in dict['columns'].items():
        if value['sdtype'] == 'numerical':
            num_score.append(RangeCoverage.compute(real_data=df1[key],
                                                      synthetic_data=df2[key]))
        elif value['sdtype'] == 'categorical':
            cat_score.append(CategoryCoverage.compute(real_data=df1[key],
                                                   synthetic_data=df2[key]))
    return np.mean(num_score), np.mean(cat_score)

def boundary(df1, df2, dict):
    score = []
    for key, value in dict['columns'].items():
        if value['sdtype'] == 'numerical':
            score.append(BoundaryAdherence.compute(real_data=df1[key],
                                                      synthetic_data=df2[key]))
    return np.mean(score)

def diversity(df1, df2, metadata):
    score = NewRowSynthesis.compute(real_data=df1, synthetic_data=df2, metadata=metadata)
    return score


def main(args):
    data_name=args.data_name
    syn_method=args.syn_method
    real_data = pd.read_csv(f'Data/{data_name}/raw/{data_name}_train.csv')
    metadata = metadata_info[data_name]
    div = 0
    for i in range(5):
        syn_data = pd.read_csv(f'Data/{data_name}/syn/{data_name}_{syn_method}{i}.csv')

        report = QualityReport()
        report.generate(real_data, syn_data, metadata, verbose=False)
        print("开始测试")
        div += diversity(real_data, syn_data, metadata)
        print(f'第{i+1}次测试完成！')

        # print('Diversity:')
        # print("All: {0:.4f}\n".format(div))

        # df_col_shape = report.get_details(property_name='Column Shapes')
        # df_col_pair_trend = report.get_details(property_name='Column Pair Trends')

        # ks, tvd = col_shape(df_col_shape)
        # print('Column Shape:')
        # print("Num: {0:.4f}, Cat: {1:.4f}\n".format(ks, tvd))
        # cts, pcs = col_pair_trend(df_col_pair_trend)
        # print('Column Pair Trend:')
        # print("Num & Num: {0:.4f}, Cat & Cat or Cat & Num: {1:.4f}\n".format(pcs, cts))
        # rc, cc = coverage(real_data, syn_data, metadata)
        # print('Coverage:')
        # print("Num: {0:.4f}, Cat: {1:.4f}\n".format(rc, cc))
        # bd = boundary(real_data, syn_data, metadata)
        # print("Boundary: {0:.4f}\n".format(bd))
        # sdm_average = (ks+tvd+pcs+cts+rc+cc) / 6
        # print('sdm_average:{0:.4f}'.format((sdm_average)))
    div_mean = div / 5
    print("Diversity: {0:.4f}".format(div_mean))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_name", type=str, default="diabetes")
    parser.add_argument("syn_method", type=str, default="smote")
    args = parser.parse_args()

    metadata_info = {"german": {'columns': {'Status of existing checking account': {'sdtype': 'categorical'},
                            'Duration in month': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                            'Credit history': {'sdtype': 'categorical'}, 'Purpose': {'sdtype': 'categorical'},
                            'Credit amount': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                            'Savings account/bonds': {'sdtype': 'categorical'},
                            'Present employment since': {'sdtype': 'categorical'},
                            'Installment rate in percentage of disposable income': {'sdtype': 'numerical',
                                                                                    'computer_representation': 'Int64'},
                            'Personal status and sex': {'sdtype': 'categorical'},
                            'Other debtors / guarantors': {'sdtype': 'categorical'},
                            'Present residence since': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                            'Property': {'sdtype': 'categorical'},
                            'Age in years': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                            'Other installment plans': {'sdtype': 'categorical'}, 'Housing': {'sdtype': 'categorical'},
                            'Number of existing credits at this bank': {'sdtype': 'numerical',
                                                                        'computer_representation': 'Int64'},
                            'Job': {'sdtype': 'categorical'},
                            'Number of people being liable to provide maintenance for': {'sdtype': 'numerical',
                                                                                         'computer_representation': 'Int64'},
                            'Telephone': {'sdtype': 'categorical'}, 'foreign worker': {'sdtype': 'categorical'},
                            'status': {'sdtype': 'categorical'}}, 'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'},
                     "diabetes": {'columns':{'Pregnancies': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                            'Glucose': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                            'BloodPressure': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                            'SkinThickness': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                            'Insulin': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                            'BMI': {'sdtype': 'numerical', 'computer_representation': 'Float32'},
                                            'DiabetesPedigreeFunction': {'sdtype': 'numerical', 'computer_representation': 'Float32'},
                                            'Age': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                            'Outcome': {'sdtype': 'categorical'}},  'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'},
                     "adult": {'columns':{'age': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                          'workclass': {'sdtype': 'categorical'},
                                          'fnlwgt': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                          'education': {'sdtype': 'categorical'},
                                          'education-num': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                          'marital-status': {'sdtype': 'categorical'},
                                          'occupation': {'sdtype': 'categorical'},
                                          'relationship': {'sdtype': 'categorical'},
                                          'race': {'sdtype': 'categorical'},
                                          'sex': {'sdtype': 'categorical'},
                                          'capital-gain': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                          'capital-loss': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                          'hours-per-week': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                                          'native-country': {'sdtype': 'categorical'},
                                          'class': {'sdtype': 'categorical'}},  'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'}
                     }

    main(args)
