import pandas as pd

def merge_features_labels(features_csv, labels_csv, output_csv):
    df_features = pd.read_csv(features_csv)
    df_labels = pd.read_csv(labels_csv)

    df_labels = df_labels.rename(columns={'CSD_refc': 'CIF Name'})
    df_merged = pd.merge(df_features, df_labels[['CIF Name', 'usable_hydrogen_storage_capacity_gcmcv2']], on='CIF Name', how='left')

    df_merged.to_csv(output_csv, index=False)
    print(f"Merged features and labels saved to {output_csv}")
    return df_merged

if __name__ == "__main__":
    features_csv = "outputs/common_mofs_with_spectral_features.csv"
    labels_csv = "data/ps_usable_hydrogen_storage_capacity_gcmcv2.csv"
    output_csv = "outputs/common_mofs_with_features_and_labels.csv"
    merge_features_labels(features_csv, labels_csv, output_csv)
