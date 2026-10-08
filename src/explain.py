import shap
import matplotlib.pyplot as plt
import pickle, duckdb

def explain_model(model, X_test, y_probs):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # 3. Global summary plot
    shap.summary_plot(shap_values, X_test)
    plt.savefig('figures/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4 & 5. Local waterfall plot for highest risk sample
    highest_risk_idx = y_probs.argmax()
    shap.plots.waterfall(shap_values[highest_risk_idx])
    plt.savefig('figures/shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import duckdb
    import pandas as pd

    # 0. Read data
    con = duckdb.connect("data/fleet.duckdb")
    features_df = con.sql("SELECT * FROM scania_features").df()
    tte_df = con.sql("SELECT * FROM scania_tte").df()
    con.close()

    with open("data/test_vehicles.pkl", "rb") as f:
        test_vehicles = pickle.load(f)
    df = features_df.merge(tte_df, on='vehicle_id', how='inner')
    X = df.drop(columns=['in_study_repair', 'vehicle_id', 'time_step', 'length_of_study_time_step',
                         'mean_time_step', 'readout_count'])
    test_mask = df['vehicle_id'].isin(test_vehicles)
    X_test = X[test_mask]

    # 2. Load model
    with open("data/fleet_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 3. Probabilities for waterfall selection
    y_probs = model.predict_proba(X_test, validate_features=False)[:, 1]

    # 4. Explain
    explain_model(model, X_test, y_probs)