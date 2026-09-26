# 🧊 POLARIS

## AI-Powered Antarctic Sea-Ice, Iceberg Trajectory & Navigation Decision Support System

> **Smart India Hackathon 2026 — SIH26059**
> **Ministry of Earth Sciences / NCPOR**
> **Theme: Transportation & Logistics**

---

## 🌍 Overview

**POLARIS** is an AI-enabled Antarctic maritime navigation decision-support system designed to help vessels understand environmental hazards, assess navigation risk, predict iceberg movement, and identify safer and more efficient navigation routes.

The system integrates **satellite-derived sea-ice concentration, ERA5 environmental conditions, iceberg trajectory prediction, risk modelling, and intelligent route optimization** into a single interactive platform.

POLARIS transforms complex Antarctic environmental information into an easy-to-understand **navigation risk map** and uses **A*** and **Ant Colony Optimization (ACO)** to generate risk-aware routes between a vessel's starting point and destination.

---

# ❗ Problem Statement

### SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System

Antarctic maritime navigation is challenging because vessels operate in an environment where sea-ice, icebergs, weather, waves, and extreme temperatures can significantly affect navigation safety and route efficiency.

Major challenges include:

* 🧊 Dynamic sea-ice conditions
* 🗻 Iceberg presence and movement
* 🌬️ Strong and changing winds
* 🌊 High wave conditions
* ❄️ Extremely low temperatures
* 🧭 Long and complex navigation routes
* 📊 Multiple environmental factors that need to be considered simultaneously

A navigation decision-support system therefore needs to combine different environmental and hazard datasets and convert them into actionable navigation intelligence.

---

# 💡 Our Solution

POLARIS addresses this challenge by integrating **sea-ice analysis, environmental risk assessment, iceberg trajectory prediction, and intelligent route optimization** into one platform.

The system follows this pipeline:

```text
NSIDC Sea-Ice Data
        +
ERA5 Environmental Data
        +
Iceberg Trajectory Prediction
        ↓
Environmental + Iceberg Risk Assessment
        ↓
Navigation Risk Grid
        ↓
A* + Ant Colony Optimization
        ↓
Risk-Aware Optimized Route
        ↓
Interactive POLARIS Dashboard
```

### How POLARIS Works

**1. Analyse the Antarctic environment**

POLARIS processes NSIDC sea-ice concentration data and ERA5 environmental conditions such as wind, temperature, and wave conditions.

**2. Predict iceberg movement**

The iceberg prediction component generates predicted iceberg positions and trajectories.

**3. Generate a navigation risk map**

Sea-ice, environmental conditions, and iceberg proximity are converted into normalized risk values and combined into a spatial navigation risk grid.

**4. Optimize the vessel route**

POLARIS uses two route optimization approaches:

* **A*** — a classical risk-aware pathfinding approach.
* **ACO** — a swarm-based optimization approach that explores alternative routes while considering distance, environmental risk, iceberg risk, detours, and route behaviour.

**5. Visualize the results**

The Streamlit dashboard brings the complete system together and allows users to visualize the Antarctic environment, risk levels, iceberg trajectories, and optimized routes.

---

# 🎯 Objectives

* Develop an integrated Antarctic navigation decision-support system.
* Analyse Antarctic sea-ice conditions.
* Incorporate atmospheric and wave conditions into navigation risk.
* Predict potential iceberg trajectories.
* Generate a spatial navigation risk map.
* Optimize vessel routes using A* and ACO.
* Balance navigation risk and route efficiency.
* Provide an interactive visualization and decision-support dashboard.

---

# 🚀 Key Features

### 🛰️ Sea-Ice Analysis

POLARIS processes Antarctic sea-ice concentration data from NSIDC to identify areas with increased sea-ice-related navigation risk.

### 🌬️ Environmental Risk Analysis

ERA5 data is used to analyse:

* Wind speed
* Temperature
* Significant wave height
* Wave direction

### 🧊 Iceberg Trajectory Prediction

The system generates predicted iceberg positions and incorporates their proximity into navigation risk.

### 🗺️ Navigation Risk Mapping

Multiple environmental factors are combined into a unified spatial risk map.

### 🧭 Dual Route Optimization

POLARIS implements:

* A* pathfinding
* Ant Colony Optimization

This allows different route optimization approaches to be evaluated on the same navigation environment.

### 📊 Interactive Dashboard

The Streamlit dashboard provides:

* Antarctic map visualization
* Sea-ice visualization
* Environmental risk visualization
* Iceberg positions
* Predicted iceberg trajectories
* Vessel and destination selection
* Optimized routes
* Route statistics
* Navigation risk analysis

---

# 🧠 System Architecture

```text
                    ┌──────────────────────┐
                    │   NSIDC Sea-Ice      │
                    │       Data           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Sea-Ice Processing  │
                    └──────────┬───────────┘
                               │
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
          ▼                                         ▼
┌──────────────────────┐                 ┌──────────────────────┐
│    ERA5 Environment  │                 │ Iceberg Prediction   │
│                      │                 │                      │
│ • Wind               │                 │ • Trajectories       │
│ • Temperature        │                 │ • Future Positions   │
│ • Wave Height        │                 │ • Iceberg Risk       │
│ • Wave Direction     │                 │                      │
└──────────┬───────────┘                 └──────────┬───────────┘
           │                                        │
           └──────────────────┬─────────────────────┘
                              ▼
                    ┌──────────────────────┐
                    │     Risk Engine      │
                    │                      │
                    │ • Ice Risk           │
                    │ • Wind Risk          │
                    │ • Wave Risk          │
                    │ • Temperature Risk   │
                    │ • Iceberg Risk       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Navigation Risk Grid │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │                      │
                    ▼                      ▼
              ┌──────────┐          ┌──────────┐
              │    A*    │          │   ACO    │
              │  Router  │          │  Router  │
              └────┬─────┘          └────┬─────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
                    ┌──────────────────────┐
                    │   Optimized Route    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ POLARIS Dashboard    │
                    │      Streamlit       │
                    └──────────────────────┘
```

---

# 🛰️ Data Sources

## NOAA / NSIDC Sea-Ice Data

POLARIS uses the **NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration Version 6 (G02202)** for Antarctic sea-ice analysis.

The dataset provides spatial sea-ice concentration information derived from passive microwave observations.

Official source:

[https://nsidc.org/data/g02202/versions/6](https://nsidc.org/data/g02202/versions/6)

---

## 🌦️ ERA5 Reanalysis

POLARIS uses **ERA5 reanalysis data** from the Copernicus Climate Data Store.

The processed dataset contains environmental information including:

* 10 m wind components
* 2 m temperature
* Significant wave height
* Mean wave direction

Official source:

[https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)

---

# ⚙️ Risk Engine

POLARIS combines environmental variables into a normalized navigation risk score.

The current environmental risk model uses:

* Sea-ice concentration
* Wind speed
* Wave height
* Temperature

The current weighting is:

```text
Environmental Risk Score =

0.55 × Ice Risk
+
0.20 × Wind Risk
+
0.15 × Wave Risk
+
0.10 × Temperature Risk
```

Iceberg proximity is then incorporated into the navigation grid to account for potential iceberg hazards.

---

# 🚨 Risk Classification

The current environmental risk score is classified into five levels:

| Risk Level  |           Score | Interpretation                        |
| ----------- | --------------: | ------------------------------------- |
| 🟢 SAFE     |        `< 0.10` | Very low estimated environmental risk |
| 🟡 LOW      | `0.10 – < 0.30` | Low estimated environmental risk      |
| 🟠 MODERATE | `0.30 – < 0.60` | Increased navigation caution          |
| 🔴 HIGH     | `0.60 – < 0.80` | Significant estimated navigation risk |
| ⚫ DANGEROUS |        `≥ 0.80` | Very high estimated navigation risk   |

---

# 🧊 Iceberg Risk

Iceberg proximity is incorporated into the navigation grid.

For each navigation-grid location, the system evaluates its distance from predicted iceberg positions.

Conceptually:

```text
Closer to predicted iceberg
          ↓
Higher iceberg risk
          ↓
Higher navigation cost
```

This allows route optimization algorithms to consider iceberg hazards alongside environmental conditions.

---

# 🧭 Route Optimization

POLARIS currently implements two route optimization methods.

## 🔵 A* Pathfinding

A* is used as a classical pathfinding baseline.

The navigation environment is represented as a geographic grid where each cell contains environmental and iceberg-related risk information.

A* evaluates possible movements between neighbouring cells and searches for a low-cost route between the vessel and destination.

The current implementation considers:

* Travel distance
* Environmental risk
* Iceberg risk
* Movement cost
* Distance to destination

The navigation grid supports 8-directional movement:

```text
↖  ↑  ↗
←  ●  →
↙  ↓  ↘
```

---

# 🟠 Ant Colony Optimization

POLARIS also implements **Ant Colony Optimization (ACO)** as an alternative route optimization technique.

ACO uses artificial ants to explore possible routes through the navigation grid.

During optimization, route quality influences pheromone reinforcement, allowing subsequent ants to favour promising paths.

The current ACO implementation considers:

* Travel distance
* Environmental risk
* Iceberg risk
* Route detours
* Directional progress
* Turning behaviour
* Pheromone information
* Destination attraction

The goal is to identify a route that balances:

```text
Route Efficiency
       +
Environmental Safety
       +
Iceberg Avoidance
```

---

# 🧠 ACO Optimization Process

```text
Initialize Navigation Grid
          ↓
Initialize Pheromone Levels
          ↓
Generate Artificial Ants
          ↓
Explore Candidate Routes
          ↓
Evaluate Route Cost
          ↓
Select Better Routes
          ↓
Update Pheromone
          ↓
Repeat Iterations
          ↓
Select Best Route
```

---

# 🗺️ Navigation Grid

The Antarctic navigation environment is converted into a discrete geographic grid.

The current dashboard uses approximately **0.5° grid spacing** for route optimization.

Each grid cell contains information such as:

```text
Latitude
Longitude
Environmental Risk
Iceberg Risk
Navigation Risk
```

The grid is then treated as a graph for route optimization.

---

# 📊 POLARIS Dashboard

The POLARIS dashboard is built using **Streamlit** and **Plotly**.

It provides an interactive interface for analysing the Antarctic navigation environment.

### Dashboard capabilities include:

* Interactive Antarctic map
* Sea-ice layer
* Environmental risk layer
* Current iceberg positions
* Predicted iceberg trajectories
* Vessel starting position
* Destination selection
* Risk weighting
* Iceberg weighting
* A* routing
* ACO routing
* Route distance
* Average navigation risk
* Maximum navigation risk
* Estimated fuel-related metric
* Route analysis

---

# 🔄 End-to-End Data Pipeline

## Step 1 — Sea-Ice Processing

```text
NSIDC NetCDF
      ↓
Dataset Inspection
      ↓
Coordinate Processing
      ↓
Antarctic Geographic Grid
      ↓
antarctic_ice.csv
```

---

## Step 2 — ERA5 Processing

```text
ERA5 NetCDF
      ↓
Dataset Processing
      ↓
Wind / Temperature / Wave Extraction
      ↓
era5_environment.csv
```

---

## Step 3 — Risk Generation

```text
antarctic_ice.csv
        +
era5_environment.csv
        ↓
Risk Engine
        ↓
Environmental Risk Score
        ↓
risk_map.csv
```

---

## Step 4 — Iceberg Prediction

```text
Iceberg Information
        ↓
Trajectory Processing
        ↓
Prediction
        ↓
Future Iceberg Positions
        ↓
iceberg_predictions.csv
```

---

## Step 5 — Navigation

```text
risk_map.csv
      +
iceberg_predictions.csv
      ↓
Navigation Grid
      ↓
 ┌───────────────┐
 │               │
 ▼               ▼
A*              ACO
 │               │
 └───────┬───────┘
         ▼
 Optimized Route
```

---

# 🛠️ Technology Stack

| Category             | Technology              |
| -------------------- | ----------------------- |
| Programming Language | Python                  |
| Dashboard            | Streamlit               |
| Visualization        | Plotly                  |
| Data Processing      | Pandas                  |
| Numerical Computing  | NumPy                   |
| Machine Learning     | Scikit-learn            |
| Sea-Ice Data         | NOAA / NSIDC            |
| Environmental Data   | ERA5                    |
| Pathfinding          | A*                      |
| Optimization         | Ant Colony Optimization |
| Data Formats         | CSV / NetCDF            |
| Version Control      | Git / GitHub            |

---

# 📁 Project Structure

```text
POLARIS/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── antarctic_ice.csv
│   ├── era5_environment.csv
│   ├── iceberg_predictions.csv
│   └── risk_map.csv
│
├── src/
│   ├── aco_router.py
│   ├── download_era5.py
│   ├── generate_data.py
│   ├── iceberg_model.py
│   ├── inspect_era5.py
│   ├── inspect_nsidc.py
│   ├── process_era5.py
│   ├── process_nsidc.py
│   └── risk_engine.py
│
├── .gitignore
└── README.md
```

---

# 🧩 Source Code Components

### `dashboard/app.py`

Main Streamlit application responsible for:

* Dashboard interface
* Mission configuration
* Map visualization
* Risk visualization
* Iceberg visualization
* Route calculation
* Route statistics
* A* routing
* ACO routing

### `src/aco_router.py`

Ant Colony Optimization route engine responsible for:

* Navigation-grid exploration
* Pheromone management
* Candidate movement selection
* Route-cost evaluation
* Detour control
* Directional progress
* Route optimization

### `src/iceberg_model.py`

Contains the iceberg trajectory prediction component.

### `src/risk_engine.py`

Combines sea-ice and environmental conditions to generate the navigation risk map.

### `src/process_nsidc.py`

Processes the NSIDC sea-ice dataset into the project's geographic CSV format.

### `src/process_era5.py`

Processes ERA5 environmental data into the project's environmental CSV format.

### `src/download_era5.py`

Used to retrieve ERA5 data through the Copernicus Climate Data Store API.

### `src/inspect_nsidc.py`

Used to inspect NSIDC NetCDF dataset structure and variables.

### `src/inspect_era5.py`

Used to inspect downloaded ERA5 datasets.

### `src/generate_data.py`

Utility script used within the project's data-generation workflow.

---

# 💻 Installation

## Requirements

Recommended:

```text
Python 3.10+
```

---

## Clone the Repository

```bash
git clone https://github.com/Inseusmohit/POLARIS.git
cd POLARIS
```

---

## Install Dependencies

Install the main dependencies:

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

For the data-processing workflow:

```bash
pip install xarray netCDF4 geopandas shapely
```

---

# ▶️ Run the Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

The POLARIS dashboard will open in your browser.

---

# 🔐 ERA5 API Configuration

The ERA5 download workflow uses the **Copernicus Climate Data Store API**.

API credentials should be configured locally according to the official CDS documentation.

**Never upload `.cdsapirc`, API keys, passwords, tokens, or other credentials to GitHub.**

---

# 📂 Processed Data

The repository currently contains the processed datasets required by the dashboard:

```text
data/
│
├── antarctic_ice.csv
├── era5_environment.csv
├── iceberg_predictions.csv
└── risk_map.csv
```

Raw NetCDF datasets and local API credentials are kept outside the GitHub repository.

---

# 🧪 Current Prototype Status

POLARIS is currently a **research and demonstration prototype**.

The current implementation uses:

* Processed NSIDC sea-ice data
* Processed ERA5 environmental data
* Iceberg trajectory predictions
* Environmental risk modelling
* A* route optimization
* Ant Colony Optimization
* Interactive Streamlit visualization

The dashboard currently operates using **locally processed datasets** rather than a continuously updating live operational data feed.

---

# ⚠️ Current Limitations

### 1. No Live Operational Data Feed

The current dashboard uses locally processed datasets.

It does not continuously ingest live maritime, weather, or satellite data.

### 2. Prototype Iceberg Prediction

The current iceberg prediction component is intended for research and demonstration.

Operational iceberg forecasting would require validated observations, ocean-current information, atmospheric forcing, uncertainty modelling, and continuous updates.

### 3. Simplified Fuel Metric

The current dashboard uses a simplified distance/risk-based fuel-related metric.

It is not a complete vessel-specific fuel-consumption model.

A production implementation would require vessel characteristics such as:

* Vessel displacement
* Engine characteristics
* Propulsion efficiency
* Vessel speed
* Hull characteristics
* Sea state
* Wind resistance
* Wave resistance
* Ice resistance

### 4. Discrete Navigation Grid

The current route optimization uses a discrete geographic grid.

Therefore, routes can contain grid-aligned movements.

Future versions can introduce continuous route optimization and advanced path smoothing.

### 5. Simplified Risk Model

The current risk model uses predefined weights.

Future versions can incorporate data-driven risk estimation and uncertainty-aware models.

---

# 🚧 Future Development

POLARIS can be extended with:

### 🌐 Real-Time Data Integration

* Near-real-time sea-ice updates
* Weather forecasts
* Ocean currents
* Wave forecasts
* Updated iceberg observations

### 🤖 Advanced AI Models

Potential future models include:

* XGBoost
* Random Forest
* LSTM
* ConvLSTM
* Transformer-based spatiotemporal models
* Physics-informed machine learning

### 🌊 Ocean Current Integration

Ocean-current information can improve both iceberg trajectory prediction and vessel route optimization.

### 🧭 Dynamic Re-Routing

Routes can be recalculated when:

* New iceberg observations arrive
* Sea-ice conditions change
* Weather conditions change
* Vessel position changes
* Navigation risk changes

### ⛽ Advanced Fuel Optimization

Future versions can replace the simplified fuel metric with a vessel-specific fuel-consumption model.

### 📡 Uncertainty-Aware Prediction

Future versions can represent uncertainty in:

* Sea-ice forecasts
* Iceberg trajectories
* Weather forecasts
* Ocean currents

This could allow POLARIS to display probability-based hazard zones rather than only single predicted trajectories.

---

# 🔮 Long-Term Vision

The long-term vision of POLARIS is to evolve into an integrated Antarctic maritime decision-support platform capable of continuously combining environmental intelligence with vessel information.

```text
Satellite Data
      +
Weather Forecasts
      +
Ocean Conditions
      +
Sea-Ice Information
      +
Iceberg Observations
      +
AI Predictions
      +
Vessel Information
      ↓
Real-Time Risk Intelligence
      ↓
Dynamic Route Optimization
      ↓
Navigation Decision Support
```

---

# 🛡️ Safety Disclaimer

POLARIS is a **research and demonstration decision-support prototype**.

Its outputs should not be used as the sole basis for real-world maritime navigation.

Actual Antarctic vessel operations require validated navigation systems, authoritative maritime information, professional navigation expertise, current environmental information, appropriate operational procedures, and compliance with applicable maritime regulations.

POLARIS demonstrates how AI, environmental data, spatial risk modelling, and optimization algorithms can support navigation decision-making.

---

# 🎓 Smart India Hackathon 2026

**Problem Statement:** SIH26059

**Organization:** Ministry of Earth Sciences / NCPOR

**Theme:** Transportation & Logistics

**Project:** POLARIS

**Focus:** AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System

---

# 👥 Project

## POLARIS

### AI-Powered Antarctic Navigation & Risk Intelligence System

Developed for **Smart India Hackathon 2026**.

---

# ⭐ POLARIS at a Glance

```text
┌───────────────────────────────────────────────┐
│                   POLARIS                     │
│                                               │
│       Antarctic Navigation Intelligence      │
│                                               │
│  🛰️  NSIDC Sea-Ice Analysis                  │
│  🌦️  ERA5 Environmental Analysis             │
│  🧊  Iceberg Trajectory Prediction            │
│  🗺️  Navigation Risk Mapping                  │
│  🔵  A* Route Optimization                    │
│  🟠  ACO Route Optimization                   │
│  📊  Interactive Streamlit Dashboard          │
│                                               │
└───────────────────────────────────────────────┘
```

> **POLARIS — Turning Antarctic environmental intelligence into navigation decisions.**

```
```
