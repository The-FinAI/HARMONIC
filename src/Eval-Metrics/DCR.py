import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics

def privacy_metrics(real_path, fake_path, data_percent=100):
    """
    Returns privacy metrics

    Inputs:
    1) real_path -> path to real data
    2) fake_path -> path to corresponding synthetic data
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics

    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets

    """

    # Loading real and synthetic datasets and removing duplicates if any
    real = pd.read_csv(real_path).drop_duplicates(keep=False)
    fake = pd.read_csv(fake_path).drop_duplicates(keep=False)

    # label_encoder
    label_encoder = LabelEncoder()
    columns = [
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
    for i in columns:
        real[i] = label_encoder.fit_transform(real[i])
        fake[i] = label_encoder.fit_transform(fake[i])

    # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
    real_sampled = real.sample(n=int(len(real) * (.01 * data_percent)), random_state=42).to_numpy()
    fake_sampled = fake.sample(n=int(len(fake) * (.01 * data_percent)), random_state=42).to_numpy()


    # Scaling real and synthetic data samples
    scalerR = StandardScaler()
    scalerR.fit(real_sampled)
    scalerF = StandardScaler()
    scalerF.fit(fake_sampled)
    df_real_scaled = scalerR.transform(real_sampled)
    df_fake_scaled = scalerF.transform(fake_sampled)

    # Computing pair-wise distances between real and synthetic
    dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within real
    dist_rr = metrics.pairwise_distances(df_real_scaled, Y=None, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within synthetic
    dist_ff = metrics.pairwise_distances(df_fake_scaled, Y=None, metric='minkowski', n_jobs=-1)

    # Removes distances of data points to themselves to avoid 0s within real and synthetic
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0], dtype=bool)].reshape(dist_rr.shape[0], -1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0], dtype=bool)].reshape(dist_ff.shape[0], -1)

    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]
    # Computing first and second smallest nearest neighbour distances within real
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    # Computing first and second smallest nearest neighbour distances within synthetic
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]

    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf, 5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr, 5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff, 5)
    nn_ratio_rf = np.array([i[0] / i[1] for i in smallest_two_rf])
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf, 5)
    nn_ratio_rr = np.array([i[0] / i[1] for i in smallest_two_rr])
    nn_fifth_perc_rr = np.percentile(nn_ratio_rr, 5)
    nn_ratio_ff = np.array([i[0] / i[1] for i in smallest_two_ff])
    nn_fifth_perc_ff = np.percentile(nn_ratio_ff, 5)


    return np.array(
        [fifth_perc_rf, fifth_perc_rr, fifth_perc_ff, nn_fifth_perc_rf, nn_fifth_perc_rr, nn_fifth_perc_ff]).reshape(1,
                                                                                                                     6)

result = privacy_metrics('../datanew/real_german_train_val.csv', '../gr/tabula/llama_tabula_sample_0.3_50.csv')
result1 = sum(result[0][:3]) / 3
print(result)
print(f'{result1:.4f}')

# df1 = pd.read_csv('/home/wangyx/relat_to_local/mydata/SyntheticData/fraud/ddpm/ddpm.csv')
# print(len(df1.columns))