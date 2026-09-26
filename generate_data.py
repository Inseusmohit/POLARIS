import numpy as np
import pandas as pd

np.random.seed(42)

# Antarctic region
latitudes = np.arange(-75, -59, 0.5)
longitudes = np.arange(-180, 181, 0.5)

rows = []

for lat in latitudes:
    for lon in longitudes:

        # Simulated Antarctic sea ice
        distance_from_pole = abs(lat + 90)

        ice = 90 - distance_from_pole * 2
        ice += np.random.normal(0, 8)

        ice = np.clip(ice, 0, 100)

        rows.append([
            lat,
            lon,
            ice
        ])

df = pd.DataFrame(
    rows,
    columns=["latitude", "longitude", "sea_ice_concentration"]
)

df.to_csv("data/antarctic_ice.csv", index=False)

print("Dataset created!")
print(df.head())
print(f"Total points: {len(df)}")