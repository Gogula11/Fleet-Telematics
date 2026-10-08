import xgboost as xgb
from sklearn.metrics import classification_report, recall_score, precision_score, average_precision_score
import duckdb
import pickle
import json
import numpy as np

def train_model(features_df, tte_df):
    df = features_df.merge(tte_df, on = 'vehicle_id', how = 'inner')

    y = df['in_study_repair']
    X = df.drop(columns=['in_study_repair', 'vehicle_id', 'time_step', 'length_of_study_time_step',
                         'mean_time_step', 'readout_count'])

    # Stratified vehicle-level split: sort vehicles by label, deal into 10 folds
    # so each split gets ~9.65% positives (prevents train/test skew)
    vehicle_labels = df.drop_duplicates('vehicle_id').set_index('vehicle_id')['in_study_repair']
    sorted_vehicles = vehicle_labels.sort_values().index
    folds = [sorted_vehicles[i::10] for i in range(10)]

    train_vehicles = np.concatenate(folds[0:7])
    val_vehicles = np.concatenate(folds[7:9])
    test_vehicles = folds[9]

    train_mask = df['vehicle_id'].isin(train_vehicles)
    val_mask = df['vehicle_id'].isin(val_vehicles)
    test_mask = df['vehicle_id'].isin(test_vehicles)

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    print("train class counts:", y_train.value_counts().to_dict())
    print("val class counts:", y_val.value_counts().to_dict())
    print("test class counts:", y_test.value_counts().to_dict())

    print("Baseline (all healthy) accuracy:", (y_test == 0).mean())

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    #train test

    from sklearn.model_selection import ParameterSampler
    from scipy.stats import randint, loguniform, uniform

    param_space = {
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.5, 0.5),
        'min_child_weight': randint(1, 11),
    }
    sampler = ParameterSampler(param_space, n_iter=30, random_state=42)

    results = []
    for i, params in enumerate(sampler):
        model = xgb.XGBClassifier(
            **params,
            tree_method='hist', device='cuda',
            scale_pos_weight=scale_pos_weight,
            random_state=42, eval_metric='logloss',
            n_estimators=1000, early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_pr = average_precision_score(
            y_val, model.predict_proba(X_val, validate_features=False)[:, 1])
        results.append({'iter': i, **params,
                        'best_iter': getattr(model, 'best_iteration', None), 'val_pr_auc': val_pr})

    best = max(results, key=lambda r: r['val_pr_auc'])
    print("Best:", best)

    model = xgb.XGBClassifier(
        **{k: v for k, v in best.items() if k in param_space},
        tree_method='hist', device='cuda',
        scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric='logloss',
        n_estimators=1000, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Threshold tuning on validation PR curve (maximize F1)
    from sklearn.metrics import precision_recall_curve
    val_probs = model.predict_proba(X_val, validate_features=False)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_threshold = thresholds[f1_scores[:-1].argmax()]
    print(f"Best threshold (val F1): {best_threshold:.3f}")
    print(f"Best iteration: {model.best_iteration}")

    probs = model.predict_proba(X_test, validate_features=False)[:, 1]
    preds = (probs >= best_threshold).astype(int)

    print(classification_report(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("PR-AUC:", average_precision_score(y_test, probs))

    metrics = {
        'recall': float(recall_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds)),
        'pr_auc': float(average_precision_score(y_test, probs)),
        'base_rate': float(y_test.mean()),
        'threshold': float(best_threshold),
        'best_iter': getattr(model, 'best_iteration', None),
        'best_params': {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in best.items() if k in param_space},
    }
    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to data/metrics.json")

    # 11. Save model for Phase 5
    with open("data/fleet_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("data/test_vehicles.pkl", "wb") as f:
        pickle.dump(test_vehicles, f)      # test_vehicles = folds[9], defined line 21

    return model


if __name__ == "__main__":
    con = duckdb.connect('data/fleet.duckdb')
    features_df = con.sql("SELECT * FROM scania_features").df()

    tte_df = con.sql("SELECT * FROM scania_tte").df()

    con.close()
    model = train_model(features_df,tte_df)
    print("Model training complete.")
