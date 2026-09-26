import cdsapi

print("Connecting to CDS...")

client = cdsapi.Client()

print("Requesting ERA5 Antarctic weather data...")

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": ["reanalysis"],

        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "significant_height_of_combined_wind_waves_and_swell",
            "mean_wave_direction"
        ],

        "year": ["2025"],
        "month": ["04"],
        "day": ["01"],

        "time": [
            "00:00",
            "06:00",
            "12:00",
            "18:00"
        ],

        # North, West, South, East
        "area": [
            -55,
            -180,
            -80,
            180
        ],

        "data_format": "netcdf"
    },

    "data/raw/era5_20250401.nc"
)

print()
print("================================")
print("ERA5 DOWNLOAD SUCCESSFUL!")
print("================================")
print("Saved:")
print("data/raw/era5_20250401.nc")