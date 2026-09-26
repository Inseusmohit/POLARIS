import numpy as np
import pandas as pd

np.random.seed(42)

icebergs = []

for iceberg_id in range(1, 11):

    lat = np.random.uniform(-70, -62)
    lon = np.random.uniform(-100, 100)

    # Random ocean drift
    velocity_lat = np.random.uniform(-0.03, 0.03)
    velocity_lon = np.random.uniform(-0.08, 0.08)

    for hour in range(49):

        predicted_lat = (
            lat + velocity_lat * hour
        )

        predicted_lon = (
            lon + velocity_lon * hour
        )

        icebergs.append([
            iceberg_id,
            hour,
            predicted_lat,
            predicted_lon
        ])


df = pd.DataFrame(
    icebergs,
    columns=[
        "iceberg_id",
        "hours_ahead",
        "latitude",
        "longitude"
    ]
)

df.to_csv(
    "data/iceberg_predictions.csv",
    index=False
)

print("Iceberg predictions generated.")
print(df.head(10))