import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

features = [
    'lap_unweighted_mean', 'lap_unweighted_median',
    'lap_unweighted_variance', 'lap_unweighted_second_largest',
    'lap_unweighted_second_smallest',
    'lap_weighted1_mean', 'lap_weighted1_median',
    'lap_weighted1_variance', 'lap_weighted1_second_largest',
    'lap_weighted1_second_smallest',
    'lap_weighted2_mean', 'lap_weighted2_median',
    'lap_weighted2_variance', 'lap_weighted2_second_largest',
    'lap_weighted2_second_smallest',
    'num_vertices', 
    'num_edges',
    'highest_degree',
    'first_zagreb_index', 'second_zagreb_index'
]

def load_data(file):
    df = pd.read_csv(file)
    df = df.drop(columns=['CIF Name'])
    return df

def main():
    df = load_data("outputs/common_mofs_with_features_and_labels.csv")
    label_column = 'usable_hydrogen_storage_capacity_gcmcv2'

    X = df[features]
    y = df[label_column]

    models = {
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": xgb.XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=200),
    }

    param_grids = {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }
    }

    results = []
    n_random_states = 100
    random_states = random.sample(range(1, 10001), n_random_states)

    for model_name, model in models.items():
        mse_list = []
        pearson_list = []
        mae_list = []

        if model_name in param_grids:
            grid = GridSearchCV(model, param_grid=param_grids[model_name], cv=3,
                                scoring="neg_mean_squared_error", n_jobs=-1)
            grid.fit(X, y)
            model = grid.best_estimator_
            print(f"Best params for {model_name}: {grid.best_params_}")

        for rs in random_states:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
            if hasattr(model, 'random_state'):
                model.set_params(random_state=rs)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            pearson_r, _ = pearsonr(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            mse_list.append(rmse)
            pearson_list.append(pearson_r)
            mae_list.append(mae)

            print(f"{model_name} Pearson r: {pearson_r:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}")

        results.append({
            "Model": model_name,
            "Avg Pearson r": np.mean(pearson_list),
            "Avg RMSE": np.mean(mse_list),
            "Avg MAE": np.mean(mae_list),
        })

    results_df = pd.DataFrame(results).sort_values("Avg Pearson r", ascending=False)
    print("\nModel Comparison:")
    print(results_df)

if __name__ == "__main__":
    main()
