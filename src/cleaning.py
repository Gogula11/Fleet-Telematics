import duckdb 
import pandas as pd

def clean_readouts(df):

    null_counts = df.isna().sum()
    print("Top Missing Columns:")
    print(null_counts[null_counts > 0].sort_values(ascending=False).head(10))

    dupes = df.duplicated(subset=['vehicle_id', 'time_step']).sum()
    print("Number of duplicate readouts:", dupes)

    counter_cols = ['171_0', '666_0', '427_0', '837_0', '309_0', '835_0', '370_0', '100_0']

    for col in df.columns:
        if df[col].isna().any():
            if col in counter_cols:
                # Vehicle-specific median for counters
                df[col] = df.groupby('vehicle_id')[col].transform(lambda x: x.fillna(x.median()))
            else:
                # Histogram bins: fill with 0 (no readings in that bin)
                df[col] = df[col].fillna(0)
    assert df.isna().sum().sum() == 0, "NaN values remain after imputation"

    return df

if __name__ == "__main__":

    con = duckdb.connect("data/fleet.duckdb")

    raw_df = con.sql("SELECT * FROM scania_readouts").df()

    clean_df = clean_readouts(raw_df)

    con.execute("CREATE OR REPLACE TABLE scania_readouts_clean as SELECT * FROM clean_df")
    print("Shape:",clean_df.shape)
    con.close()