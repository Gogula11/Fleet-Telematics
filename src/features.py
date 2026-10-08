import numpy as np
import pandas as pd
import duckdb

def generate_features(df):
    df = df.sort_values(['vehicle_id', 'time_step'])
    counter_cols = ['171_0', '666_0', '427_0', '837_0', '309_0', '835_0', '370_0', '100_0']

    for col in counter_cols:
        df[col + '_rolling_mean'] = df.groupby('vehicle_id')[col].rolling(5).mean().reset_index(0, drop=True)
        df[col + '_rolling_std'] = df.groupby('vehicle_id')[col].rolling(5).std().reset_index(0, drop=True)
        
        
        df[col + '_delta'] = df.groupby('vehicle_id')[col].diff()


    df['readout_count'] = df.groupby('vehicle_id')['time_step'].transform('count')
    df['mean_time_step'] = df.groupby('vehicle_id')['time_step'].transform('mean')

    return df

def generate_histogram_features(df):
    groups = {
        '167': [f'167_{i}' for i in range(10)],
        '272': [f'272_{i}' for i in range(10)],
        '291': [f'291_{i}' for i in range(11)],
        '158': [f'158_{i}' for i in range(10)],
        '459': [f'459_{i}' for i in range(20)],
        '397': [f'397_{i}' for i in range(36)],
    }

    for name, cols in groups.items():
        vals = df[cols].values
        indices = np.array([int(c.split('_')[1]) for c in cols])

        total = vals.sum(axis=1)
        total[total == 0] = 1

        mean_index = (vals * indices).sum(axis=1) / total
        df[name + '_mean_index'] = mean_index

        p = vals / total[:, None]
        p[p == 0] = 1e-12
        df[name + '_entropy'] = -(p * np.log(p)).sum(axis=1)

        df[name + '_max_bin'] = vals.argmax(axis=1)

    for name in groups:
        col = name + '_mean_index'
        df[col + '_rolling_mean'] = df.groupby('vehicle_id')[col].rolling(5).mean().reset_index(0, drop=True)
        df[col + '_rolling_std'] = df.groupby('vehicle_id')[col].rolling(5).std().reset_index(0, drop=True)
        df[col + '_delta'] = df.groupby('vehicle_id')[col].diff()

    return df

if __name__ == "__main__":
    con = duckdb.connect('data/fleet.duckdb')

    clean_df = con.sql("SELECT * FROM scania_readouts_clean").df()

    feature_df = generate_features(clean_df)
    feature_df = generate_histogram_features(feature_df)

    con.execute("CREATE OR REPLACE TABLE scania_features AS SELECT * FROM feature_df")

    print("Shape:",feature_df.shape)
    con.close()