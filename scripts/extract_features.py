import os
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

def compute_laplacian(adj_matrix):
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian_matrix = degree_matrix - adj_matrix
    return laplacian_matrix

def spectral_features(eigenvalues):
    eigenvalues = np.round(eigenvalues, 4)
    eigenvalues_sorted = np.sort(eigenvalues)
    positive_eigenvalues = eigenvalues_sorted[eigenvalues_sorted > 0]
    features = {
        'mean': np.round(np.mean(eigenvalues), 4),
        'median': np.round(np.median(eigenvalues), 4),
        'variance': np.round(np.var(eigenvalues), 4),
        'max': np.round(np.max(eigenvalues), 4),
        'second_largest': np.round(eigenvalues_sorted[-2], 4) if len(eigenvalues_sorted) > 1 else None,
        'min': np.round(np.min(eigenvalues), 4),
        'second_smallest': np.round(positive_eigenvalues[1], 4) if len(positive_eigenvalues) > 1 else None,
    }
    return features

def main(csv_file, output_csv):
    df_csv = pd.read_csv(csv_file)
    df_csv = df_csv.drop_duplicates(subset='CIF Name', keep='first')
    cif_names = df_csv['CIF Name']
    
    results = []
    count = 0
    
    for cif_name in cif_names:
        cif_name_lower = cif_name.lower()
        files = [f for f in os.listdir('data') if f.lower().startswith(cif_name_lower + "_") and (
            f.lower().endswith('_unweighted.npy') or 
            f.lower().endswith('_adjacency_matrix.npy') or 
            f.lower().endswith('_sigmoid_distance.npy')
        )]

        if len(files) < 3:
            print(f"{count+1} Missing or extra files for {cif_name}. Skipping.")
            continue
        
        unweighted_file = next(f for f in files if f.lower().endswith('_unweighted.npy'))
        weighted_file1 = next(f for f in files if f.lower().endswith('_adjacency_matrix.npy'))
        weighted_file2 = next(f for f in files if f.lower().endswith('_sigmoid_distance.npy'))
        
        unweighted_adj = np.load(os.path.join('data', unweighted_file))
        weighted_adj1 = np.load(os.path.join('data', weighted_file1))
        weighted_adj2 = np.load(os.path.join('data', weighted_file2))
        
        if unweighted_adj.shape[0] == 0:
            continue
        
        lap_unweighted = compute_laplacian(np.round(unweighted_adj,4))
        lap_weighted1 = compute_laplacian(weighted_adj1)
        lap_weighted2 = compute_laplacian(weighted_adj2)
        
        eigvals_unweighted = eigvalsh(unweighted_adj)
        eigvals_weighted1 = eigvalsh(weighted_adj1)
        eigvals_weighted2 = eigvalsh(weighted_adj2)
        
        eigvals_lap_unweighted = eigvalsh(lap_unweighted)
        eigvals_lap_weighted1 = eigvalsh(lap_weighted1)
        eigvals_lap_weighted2 = eigvalsh(lap_weighted2)
        
        positive_unweighted = eigvals_unweighted[eigvals_unweighted > 0]
        positive_weighted1 = eigvals_weighted1[eigvals_weighted1 > 0]
        positive_weighted2 = eigvals_weighted2[eigvals_weighted2 > 0]

        lap_unweighted_features = spectral_features(eigvals_lap_unweighted)
        lap_weighted1_features = spectral_features(eigvals_lap_weighted1)
        lap_weighted2_features = spectral_features(eigvals_lap_weighted2)
        unweighted_adj_features = spectral_features(eigvals_unweighted)
        weighted1_adj_features = spectral_features(eigvals_weighted1)
        weighted2_adj_features = spectral_features(eigvals_weighted2)
        positive_unweighted_adj_features = spectral_features(positive_unweighted)
        positive_weighted1_adj_features = spectral_features(positive_weighted1)
        positive_weighted2_adj_features = spectral_features(positive_weighted2)
        
        result = {
            'CIF Name': cif_name,
            'lap_unweighted': lap_unweighted_features,
            'lap_weighted1': lap_weighted1_features,
            'lap_weighted2': lap_weighted2_features,
            'unweighted_adj': unweighted_adj_features,
            'weighted1_adj': weighted1_adj_features,
            'weighted2_adj': weighted2_adj_features,
            'positive_unweighted_adj': positive_unweighted_adj_features,
            'positive_weighted1_adj': positive_weighted1_adj_features,
            'positive_weighted2_adj': positive_weighted2_adj_features,
        }
        
        flat_result = {key + "_" + feature: value for key, features in result.items() if key != 'CIF Name' for feature, value in features.items()}
        flat_result['CIF Name'] = cif_name
        results.append(flat_result)
        count += 1
        print(f"{count} processed {cif_name}")
        
    df = pd.DataFrame(results)
    cols = ['CIF Name'] + [col for col in df.columns if col != 'CIF Name']
    df = df[cols]
    
    final_df = df.merge(df_csv[['CIF Name', 'UV at PS', 'UG at PS']], on='CIF Name', how='left')
    final_df = final_df[cols]
    
    final_df.to_csv(output_csv, index=False)
    print(f"Saved spectral features to {output_csv}")
    return final_df

if __name__ == "__main__":
    csv_file = "data/common_mofs_lower.csv"
    output_csv = "outputs/common_mofs_with_spectral_features.csv"
    main(csv_file, output_csv)
