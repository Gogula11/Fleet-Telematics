import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import duckdb
import json
import pickle
import pandas as pd
import numpy as np


def main():
    con = duckdb.connect("data/fleet.duckdb")
    features_df = con.sql("SELECT * FROM scania_features").df()
    tte_df = con.sql("SELECT vehicle_id, in_study_repair FROM scania_tte").df()
    con.close()

    with open("data/fleet_model.pkl", "rb") as f:
        model = pickle.load(f)

    df = features_df.merge(tte_df, on='vehicle_id', how='inner')
    y = df['in_study_repair']
    X = df.drop(columns=['in_study_repair', 'vehicle_id', 'time_step',
                         'mean_time_step', 'readout_count'])
    with open("data/test_vehicles.pkl", "rb") as f:
        test_vehicles = pickle.load(f)
    test_mask = df['vehicle_id'].isin(test_vehicles)
    X_test = X[test_mask]
    y_test = y[test_mask]

    probs = model.predict_proba(X_test, validate_features=False)[:, 1]
    test_df = df[test_mask].copy()
    test_df['risk'] = probs

    # 1a. Fleet risk bands (standalone figure)
    truck_risk = test_df.groupby('vehicle_id').agg(
        mean_risk=('risk', 'mean'), label=('in_study_repair', 'first')
    )

    high = truck_risk[truck_risk['mean_risk'] > 0.7]
    medium = truck_risk[(truck_risk['mean_risk'] > 0.4) & (truck_risk['mean_risk'] <= 0.7)]
    low = truck_risk[truck_risk['mean_risk'] <= 0.4]

    bands = ['High\n(>70%)', 'Medium\n(40-70%)', 'Low\n(<40%)']
    counts = [len(high), len(medium), len(low)]
    pcts = [c / len(truck_risk) * 100 for c in counts]
    failed_per_band = [int(b['label'].sum()) for b in (high, medium, low)]
    band_colours = ['red', 'orange', 'green']
    plt.figure(figsize=(10, 5))
    bars = plt.barh(bands, pcts, color=band_colours)
    plt.gca().invert_yaxis()
    plt.bar_label(bars, labels=[
        f'{c} ({p:.1f}%) — {f} actually failed'
        for c, p, f in zip(counts, pcts, failed_per_band)
    ], padding=3)
    plt.xlabel('% of test fleet')
    plt.xlim(0, 100)
    plt.title('Fleet Risk Bands')
    plt.tight_layout()
    plt.savefig('figures/risk_bands.png', dpi=150)
    plt.close()

    # 1b. Top 40 trucks to inspect (standalone figure)
    top20 = truck_risk.sort_values('mean_risk', ascending=False).head(40)

    plt.figure(figsize=(10, 12))
    bar_colours = ['red' if l == 1 else 'lightcoral' for l in top20['label']]
    bars2 = plt.barh(range(len(top20)), top20['mean_risk'] * 100, color=bar_colours, edgecolor='black', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.yticks(range(len(top20)), [f'Truck {v}' for v in top20.index])
    plt.bar_label(bars2, fmt='%.0f%%', padding=3)
    plt.xlabel('Mean failure risk (%)')
    plt.title('Top 40 Trucks to Inspect\n(dark red = actually failed, light red = flagged but healthy)')
    plt.xlim(top20['mean_risk'].min() * 100 - 2, 100)
    plt.tight_layout()
    plt.savefig('figures/top_trucks.png', dpi=150)
    plt.close()

    # 2. Risk timeline
    vehicle_risk = test_df.groupby('vehicle_id').agg(
        max_risk=('risk', 'max'), label=('in_study_repair', 'first')
    ).sort_values('max_risk', ascending=False)

    failed = vehicle_risk[vehicle_risk['label'] == 1]
    healthy = vehicle_risk[vehicle_risk['label'] == 0]
    n_failed = min(6, len(failed))
    n_healthy = 18 - n_failed
    picked = pd.concat([
        failed.head(n_failed),
        healthy.iloc[np.linspace(0, len(healthy) - 1, n_healthy).astype(int)]
    ])

    plt.figure(figsize=(12, 7))
    for _, row in picked.iterrows():
        truck = row.name
        tdf = test_df[test_df['vehicle_id'] == truck].sort_values('time_step')
        x = np.arange(len(tdf))
        colour = 'red' if row['label'] == 1 else 'green'
        alpha = 0.9 if row['label'] == 1 else 0.5
        plt.plot(x, tdf['risk'] * 100, color=colour, alpha=alpha, linewidth=1.5)
    plt.xlabel('Readout number (per truck)')
    plt.ylabel('Failure risk (%)')
    plt.title('Risk Score Over Time — 18 Trucks (Red = Failed, Green = Healthy)')
    plt.ylim(0, 100)
    from matplotlib.lines import Line2D
    plt.legend(handles=[
        Line2D([0], [0], color='red', label='Failed truck'),
        Line2D([0], [0], color='green', label='Healthy truck'),
    ])
    plt.tight_layout()
    plt.savefig('figures/risk_timeline.png', dpi=150)
    plt.close()

    # 3. Inspection value
    with open("data/metrics.json") as f:
        m = json.load(f)
    plt.figure(figsize=(8, 6))
    plt.bar(['Random\ninspection', 'Model-guided\ninspection'],
            [m['base_rate'] * 100, m['precision'] * 100], color=['grey', 'green'])
    plt.ylabel('Real failures found per 100 inspections')
    plt.title(f"Why Use the Model? — {m['precision'] / m['base_rate']:.1f}x More Failures Caught")
    plt.tight_layout()
    plt.savefig('figures/inspection_value.png', dpi=150)
    plt.close()

    # 4. Metrics card
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    lines = [
        'MODEL RESULTS — PLAIN SUMMARY',
        '',
        f"• Catches {round(m['recall'] * 100)} of 100 breakdowns before they happen",
        f"• {m['precision'] / m['base_rate']:.1f}x better than random inspection",
        f"• Ranks trucks {m['pr_auc'] / m['base_rate']:.1f}x better than chance",
        f"• {round(m['precision'] * 100)}% of model alarms are real failures",
        '',
        'Data: 1.1M sensor readouts, 23,550 SCANIA trucks',
    ]
    plt.text(0.1, 0.5, '\n'.join(lines), fontsize=14, verticalalignment='center', family='monospace')
    plt.savefig('figures/metrics_card.png', dpi=150)
    plt.close()

    print("7 figures saved to figures/")


if __name__ == "__main__":
    main()
