import pandas as pd
import numpy as np

print("==========================================")
print("POLARIS REAL ENVIRONMENTAL RISK ENGINE")
print("==========================================")

# ============================================================
# 1. LOAD NSIDC SEA-ICE DATA
# ============================================================

print("\nLoading NSIDC sea-ice data...")

ice = pd.read_csv(
    "data/antarctic_ice.csv"
)

ice = ice.dropna(
    subset=[
        "latitude",
        "longitude",
        "sea_ice_concentration"
    ]
)

print("NSIDC records:", len(ice))


# ============================================================
# 2. LOAD ERA5 ENVIRONMENT DATA
# ============================================================

print("\nLoading ERA5 environmental data...")

era5 = pd.read_csv(
    "data/era5_environment.csv"
)

era5 = era5.dropna(
    subset=[
        "latitude",
        "longitude",
        "wind_speed",
        "temperature"
    ]
)

print("ERA5 records:", len(era5))


# ============================================================
# 3. CREATE GRID FOR RISK CALCULATION
# ============================================================

print("\nCreating environmental risk grid...")


# Use ERA5 wind/temperature grid as the base
# and interpolate sea-ice concentration onto it.

ice_points = ice[
    [
        "latitude",
        "longitude",
        "sea_ice_concentration"
    ]
].copy()


# ============================================================
# 4. NEAREST-NEIGHBOUR MATCHING
# ============================================================

print("Matching NSIDC sea ice with ERA5 locations...")


# Round coordinates to make matching easier
ice_points["lat_round"] = (
    ice_points["latitude"] * 4
).round() / 4

ice_points["lon_round"] = (
    ice_points["longitude"] * 4
).round() / 4


era5["lat_round"] = (
    era5["latitude"] * 4
).round() / 4

era5["lon_round"] = (
    era5["longitude"] * 4
).round() / 4


# Aggregate NSIDC values where multiple points
# fall into the same ERA5 grid cell.

ice_grid = (
    ice_points
    .groupby(
        ["lat_round", "lon_round"],
        as_index=False
    )["sea_ice_concentration"]
    .mean()
)


# ============================================================
# 5. MERGE ICE + ERA5
# ============================================================

risk = era5.merge(
    ice_grid,
    on=["lat_round", "lon_round"],
    how="left"
)


# ============================================================
# 6. HANDLE MISSING SEA-ICE VALUES
# ============================================================

risk["sea_ice_concentration"] = (
    risk["sea_ice_concentration"]
    .fillna(0)
)


# ============================================================
# 7. ICE RISK
# ============================================================

risk["ice_risk"] = (
    risk["sea_ice_concentration"] / 100
)


# ============================================================
# 8. WIND RISK
# ============================================================

# < 8 m/s      → low
# 8–15 m/s     → increasing
# 15–25 m/s    → high
# > 25 m/s     → extreme

risk["wind_risk"] = np.clip(
    (risk["wind_speed"] - 8) / 17,
    0,
    1
)


# ============================================================
# 9. WAVE RISK
# ============================================================

# < 2 m        → low
# 2–6 m        → increasing
# > 6 m        → high

risk["wave_risk"] = np.clip(
    (risk["wave_height"].fillna(0) - 2) / 4,
    0,
    1
)


# ============================================================
# 10. TEMPERATURE RISK
# ============================================================

# Very low temperatures increase operational risk.

risk["temperature_risk"] = np.clip(
    (-risk["temperature"] - 10) / 40,
    0,
    1
)


# ============================================================
# 11. COMBINED ENVIRONMENTAL RISK
# ============================================================

risk["risk_score"] = (
    0.55 * risk["ice_risk"]
    + 0.20 * risk["wind_risk"]
    + 0.15 * risk["wave_risk"]
    + 0.10 * risk["temperature_risk"]
)


# ============================================================
# 12. RISK CLASSIFICATION
# ============================================================

def classify_risk(score):

    if score < 0.10:
        return "SAFE"

    elif score < 0.30:
        return "LOW"

    elif score < 0.60:
        return "MODERATE"

    elif score < 0.80:
        return "HIGH"

    else:
        return "DANGEROUS"


risk["risk_level"] = (
    risk["risk_score"]
    .apply(classify_risk)
)


# ============================================================
# 13. CLEAN OUTPUT
# ============================================================

output_columns = [
    "latitude",
    "longitude",
    "wind_speed",
    "temperature",
    "wave_height",
    "sea_ice_concentration",
    "ice_risk",
    "wind_risk",
    "wave_risk",
    "temperature_risk",
    "risk_score",
    "risk_level"
]

risk = risk[output_columns]


# ============================================================
# 14. SAVE
# ============================================================

output = "data/risk_map.csv"

risk.to_csv(
    output,
    index=False
)


# ============================================================
# 15. REPORT
# ============================================================

print("\n==========================================")
print("REAL POLARIS RISK MAP CREATED!")
print("==========================================")

print("\nOutput:")
print(output)

print("\nRecords:")
print(len(risk))

print("\nColumns:")
print(risk.columns.tolist())

print("\nRisk statistics:")
print(
    risk["risk_score"].describe()
)

print("\nRisk distribution:")
print(
    risk["risk_level"].value_counts()
)

print("\nAverage environmental conditions:")

print(
    "Wind:",
    round(risk["wind_speed"].mean(), 2),
    "m/s"
)

print(
    "Temperature:",
    round(risk["temperature"].mean(), 2),
    "°C"
)

print(
    "Wave height:",
    round(risk["wave_height"].mean(), 2),
    "m"
)

print(
    "Sea ice:",
    round(
        risk["sea_ice_concentration"].mean(),
        2
    ),
    "%"
)

print("\nDONE!")