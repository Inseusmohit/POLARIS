# ============================================================
# POLARIS
# AI-Powered Antarctic Navigation Decision Support System
# ============================================================

import os
import sys
import heapq
import math

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# IMPORT ACO ROUTER
# ============================================================

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from aco_router import ACOGridRouter


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="POLARIS | Antarctic Navigation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #07131f;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    .polaris-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
        color: white;
    }

    .polaris-subtitle {
        color: #9db4c7;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .status-box {
        padding: 12px 16px;
        border-radius: 8px;
        background: #0d2233;
        border: 1px solid #1b4058;
        color: #b9d8ea;
        margin-bottom: 15px;
    }

    .metric-card {
        background: #0c1d2c;
        border: 1px solid #18384e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    .metric-title {
        color: #8ca9bb;
        font-size: 0.8rem;
    }

    .metric-value {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ICE_FILE = os.path.join(BASE_DIR, "data", "antarctic_ice.csv")
RISK_FILE = os.path.join(BASE_DIR, "data", "risk_map.csv")
ICEBERG_FILE = os.path.join(BASE_DIR, "data", "iceberg_predictions.csv")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    ice = pd.read_csv(ICE_FILE)
    risk = pd.read_csv(RISK_FILE)
    icebergs = pd.read_csv(ICEBERG_FILE)

    return ice, risk, icebergs


try:
    ice, risk, icebergs = load_data()

except Exception as e:

    st.error("Unable to load POLARIS data files.")

    st.code(
        f"""
Expected files:

{ICE_FILE}
{RISK_FILE}
{ICEBERG_FILE}

Error:
{e}
"""
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

ice = ice.dropna(
    subset=[
        "latitude",
        "longitude",
        "sea_ice_concentration",
    ]
).copy()

risk = risk.dropna(
    subset=[
        "latitude",
        "longitude",
        "risk_score",
    ]
).copy()

icebergs = icebergs.dropna(
    subset=[
        "latitude",
        "longitude",
    ]
).copy()


# Keep only Antarctic region
ice = ice[ice["latitude"] <= -55].copy()
risk = risk[risk["latitude"] <= -55].copy()
icebergs = icebergs[icebergs["latitude"] <= -55].copy()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🚢 POLARIS Mission Control")

st.sidebar.markdown("---")

st.sidebar.markdown("### Mission Parameters")

vessel_name = st.sidebar.text_input(
    "Vessel",
    value="Research Vessel POLARIS",
)

st.sidebar.markdown("#### Start Position")

start_lat = st.sidebar.number_input(
    "Start latitude",
    value=-60.0,
    min_value=-89.0,
    max_value=-55.0,
    step=0.5,
)

start_lon = st.sidebar.number_input(
    "Start longitude",
    value=0.0,
    min_value=-180.0,
    max_value=180.0,
    step=0.5,
)

st.sidebar.markdown("#### Destination")

goal_lat = st.sidebar.number_input(
    "Destination latitude",
    value=-70.0,
    min_value=-89.0,
    max_value=-55.0,
    step=0.5,
)

goal_lon = st.sidebar.number_input(
    "Destination longitude",
    value=30.0,
    min_value=-180.0,
    max_value=180.0,
    step=0.5,
)


# ============================================================
# NAVIGATION SETTINGS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation AI")

risk_weight = st.sidebar.slider(
    "Risk avoidance",
    min_value=1.0,
    max_value=30.0,
    value=15.0,
    step=1.0,
)

iceberg_weight = st.sidebar.slider(
    "Iceberg avoidance",
    min_value=1.0,
    max_value=30.0,
    value=12.0,
    step=1.0,
)


# ============================================================
# ROUTING ALGORITHM
# ============================================================

st.sidebar.markdown("#### Routing Algorithm")

routing_algorithm = st.sidebar.selectbox(
    "Navigation algorithm",
    [
        "A*",
        "ACO",
        "Compare A* vs ACO",
    ],
    index=0,
)

aco_ants = 25
aco_iterations = 30
aco_alpha = 1.0
aco_beta = 4.0
aco_evaporation = 0.35

if routing_algorithm in ["ACO", "Compare A* vs ACO"]:

    st.sidebar.markdown("#### ACO Parameters")

    aco_ants = st.sidebar.slider(
        "Number of ants",
        min_value=10,
        max_value=50,
        value=25,
        step=5,
    )

    aco_iterations = st.sidebar.slider(
        "ACO iterations",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
    )

    aco_alpha = st.sidebar.slider(
        "Pheromone influence (α)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5,
    )

    aco_beta = st.sidebar.slider(
        "Heuristic influence (β)",
        min_value=1.0,
        max_value=6.0,
        value=4.0,
        step=0.5,
    )

    aco_evaporation = st.sidebar.slider(
        "Pheromone evaporation",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05,
    )


# ============================================================
# MAP LAYERS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### Map Layers")

show_sea_ice = st.sidebar.checkbox(
    "🧊 Sea ice concentration",
    value=True,
)

show_risk = st.sidebar.checkbox(
    "⚠️ Environmental risk",
    value=True,
)

show_icebergs = st.sidebar.checkbox(
    "🧊 Current iceberg positions",
    value=True,
)

show_trajectories = st.sidebar.checkbox(
    "➡️ Iceberg trajectories",
    value=True,
)

show_route = st.sidebar.checkbox(
    "🟢 Optimized route",
    value=True,
)


trajectory_hours = st.sidebar.slider(
    "Trajectory horizon (hours)",
    min_value=0,
    max_value=int(
        icebergs["hours_ahead"].max()
        if "hours_ahead" in icebergs.columns
        else 48
    ),
    value=int(
        icebergs["hours_ahead"].max()
        if "hours_ahead" in icebergs.columns
        else 48
    ),
    step=6,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="polaris-title">🧊 POLARIS</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="polaris-subtitle">'
    "AI-Powered Antarctic Sea-Ice, Iceberg Trajectory & "
    "Navigation Decision Support System"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="status-box">
    🛰️ <b>Operational Antarctic Navigation View</b>
    &nbsp; | &nbsp;
    NSIDC Sea Ice + ERA5 Environment + AI Iceberg Prediction + A* / ACO Routing
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def normalize_longitude(lon):

    while lon > 180:
        lon -= 360

    while lon < -180:
        lon += 360

    return lon


def grid_round(value):

    # Round to nearest 0.5 degree.
    return np.floor(value * 2 + 0.5) / 2


# ============================================================
# CREATE NAVIGATION GRID
# ============================================================

@st.cache_data
def prepare_navigation_grid(risk_df, iceberg_df):

    df = risk_df.copy()

    # --------------------------------------------------------
    # Create common 0.5° navigation grid
    # --------------------------------------------------------

    df["grid_lat"] = grid_round(df["latitude"])
    df["grid_lon"] = grid_round(df["longitude"])

    aggregation = {
        "risk_score": "mean",
        "ice_risk": "mean",
        "wind_risk": "mean",
        "wave_risk": "mean",
        "temperature_risk": "mean",
        "sea_ice_concentration": "mean",
        "wind_speed": "mean",
        "temperature": "mean",
        "wave_height": "mean",
    }

    available = {
        key: value
        for key, value in aggregation.items()
        if key in df.columns
    }

    grid = (
        df.groupby(
            ["grid_lat", "grid_lon"],
            as_index=False,
        )
        .agg(available)
    )

    # --------------------------------------------------------
    # Prepare iceberg prediction points
    # --------------------------------------------------------

    ib = iceberg_df.copy()

    if "hours_ahead" in ib.columns:

        ib = ib[
            ib["hours_ahead"] <= iceberg_df["hours_ahead"].max()
        ]

    iceberg_lat = ib["latitude"].to_numpy()
    iceberg_lon = ib["longitude"].to_numpy()

    # --------------------------------------------------------
    # Vectorized iceberg proximity calculation
    #
    # We process chunks to prevent huge memory usage.
    # --------------------------------------------------------

    grid_lat = grid["grid_lat"].to_numpy()
    grid_lon = grid["grid_lon"].to_numpy()

    min_distance = np.full(
        len(grid),
        9999.0,
        dtype=float,
    )

    chunk_size = 1500

    for start in range(
        0,
        len(grid),
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            len(grid),
        )

        glat = grid_lat[start:end][:, None]
        glon = grid_lon[start:end][:, None]

        # Approximate local distance in km.
        dy = (
            glat - iceberg_lat[None, :]
        ) * 111.32

        dx = (
            glon - iceberg_lon[None, :]
        ) * 111.32 * np.cos(
            np.radians(glat)
        )

        distance = np.sqrt(
            dx * dx + dy * dy
        )

        min_distance[start:end] = np.min(
            distance,
            axis=1,
        )

    # --------------------------------------------------------
    # Convert distance into iceberg risk
    #
    # 0 km  -> risk 1
    # 60 km -> risk 0
    # --------------------------------------------------------

    grid["iceberg_distance_km"] = min_distance

    grid["iceberg_risk"] = np.clip(
        1.0 - (
            grid["iceberg_distance_km"] / 60.0
        ),
        0,
        1,
    )

    # --------------------------------------------------------
    # Combined navigation risk
    # --------------------------------------------------------

    base_risk = (
        grid["risk_score"]
        if "risk_score" in grid.columns
        else 0
    )

    grid["navigation_risk"] = np.clip(
        0.75 * base_risk
        + 0.25 * grid["iceberg_risk"],
        0,
        1,
    )

    return grid


navigation_grid = prepare_navigation_grid(
    risk,
    icebergs,
)


# ============================================================
# A* ROUTING
# ============================================================

def nearest_grid_node(
    latitude,
    longitude,
    grid_nodes,
):

    best_node = None
    best_distance = float("inf")

    for node in grid_nodes:

        lat, lon = node

        distance = (
            (lat - latitude) ** 2
            + (lon - longitude) ** 2
        )

        if distance < best_distance:

            best_distance = distance
            best_node = node

    return best_node


def build_risk_lookup(
    grid,
    risk_weight_value,
    iceberg_weight_value,
):

    lookup = {}

    for row in grid.itertuples():

        base = float(
            getattr(
                row,
                "risk_score",
                0.0,
            )
        )

        iceberg = float(
            getattr(
                row,
                "iceberg_risk",
                0.0,
            )
        )

        combined = np.clip(
            base * risk_weight_value / 15.0
            + iceberg * iceberg_weight_value / 12.0,
            0,
            30,
        )

        lookup[
            (
                float(row.grid_lat),
                float(row.grid_lon),
            )
        ] = combined

    return lookup


def get_neighbors(
    node,
    node_set,
):

    lat, lon = node

    # 8-direction movement
    directions = [
        (-0.5, 0),
        (0.5, 0),
        (0, -0.5),
        (0, 0.5),
        (-0.5, -0.5),
        (-0.5, 0.5),
        (0.5, -0.5),
        (0.5, 0.5),
    ]

    neighbors = []

    for dlat, dlon in directions:

        new_lat = lat + dlat
        new_lon = normalize_longitude(
            lon + dlon
        )

        if new_lat < -90 or new_lat > -55:
            continue

        candidate = (
            round(new_lat, 1),
            round(new_lon, 1),
        )

        # The grid uses 0.5 degree positions.
        candidate = (
            round(
                math.floor(candidate[0] * 2 + 0.5)
                / 2,
                1,
            ),
            round(
                math.floor(candidate[1] * 2 + 0.5)
                / 2,
                1,
            ),
        )

        if candidate in node_set:
            neighbors.append(candidate)

    return neighbors


def a_star_route(
    grid,
    start_latitude,
    start_longitude,
    goal_latitude,
    goal_longitude,
    risk_weight_value,
    iceberg_weight_value,
):

    risk_lookup = build_risk_lookup(
        grid,
        risk_weight_value,
        iceberg_weight_value,
    )

    node_set = set(risk_lookup.keys())

    if not node_set:
        return []

    start = nearest_grid_node(
        start_latitude,
        start_longitude,
        node_set,
    )

    goal = nearest_grid_node(
        goal_latitude,
        goal_longitude,
        node_set,
    )

    if start is None or goal is None:
        return []

    # --------------------------------------------------------
    # A* heuristic
    # --------------------------------------------------------

    def heuristic(node):

        return haversine_km(
            node[0],
            node[1],
            goal[0],
            goal[1],
        )

    # --------------------------------------------------------
    # Priority queue
    # --------------------------------------------------------

    open_set = []

    heapq.heappush(
        open_set,
        (
            heuristic(start),
            0,
            start,
        ),
    )

    came_from = {}

    g_score = {
        start: 0.0
    }

    visited = set()

    while open_set:

        _, current_cost, current = heapq.heappop(
            open_set
        )

        if current in visited:
            continue

        visited.add(current)

        if current == goal:

            route = [current]

            while current in came_from:

                current = came_from[current]

                route.append(current)

            route.reverse()

            return route

        for neighbor in get_neighbors(
            current,
            node_set,
        ):

            if neighbor in visited:
                continue

            distance = haversine_km(
                current[0],
                current[1],
                neighbor[0],
                neighbor[1],
            )

            risk_cost = (
                1.0
                + risk_lookup.get(
                    neighbor,
                    0.0,
                )
            )

            movement_cost = (
                distance * risk_cost
            )

            tentative = (
                g_score[current]
                + movement_cost
            )

            if tentative < g_score.get(
                neighbor,
                float("inf"),
            ):

                came_from[neighbor] = current

                g_score[neighbor] = tentative

                f_score = (
                    tentative
                    + heuristic(neighbor)
                )

                heapq.heappush(
                    open_set,
                    (
                        f_score,
                        tentative,
                        neighbor,
                    ),
                )

    return []


# ============================================================
# CALCULATE ROUTE
# ============================================================

@st.cache_data
def calculate_route_cached(
    grid,
    start_latitude,
    start_longitude,
    goal_latitude,
    goal_longitude,
    risk_weight_value,
    iceberg_weight_value,
):

    return a_star_route(
        grid,
        start_latitude,
        start_longitude,
        goal_latitude,
        goal_longitude,
        risk_weight_value,
        iceberg_weight_value,
    )


# Convert dataframe to hashable-ish representation for cache
grid_for_route = navigation_grid.copy()

astar_route = calculate_route_cached(
    grid_for_route,
    start_lat,
    start_lon,
    goal_lat,
    goal_lon,
    risk_weight,
    iceberg_weight,
)


def build_aco_corridor(grid, start_latitude, start_longitude, goal_latitude, goal_longitude, margin=5.0):
    """Reduce the ACO search space to a corridor around the mission."""

    min_lat = min(start_latitude, goal_latitude) - margin
    max_lat = max(start_latitude, goal_latitude) + margin

    # Handle normal Antarctic longitude missions.
    min_lon = min(start_longitude, goal_longitude) - margin
    max_lon = max(start_longitude, goal_longitude) + margin

    if min_lon >= -180 and max_lon <= 180:
        corridor = grid[
            (grid["grid_lat"] >= min_lat)
            & (grid["grid_lat"] <= max_lat)
            & (grid["grid_lon"] >= min_lon)
            & (grid["grid_lon"] <= max_lon)
        ].copy()
    else:
        corridor = grid[
            (grid["grid_lat"] >= min_lat)
            & (grid["grid_lat"] <= max_lat)
        ].copy()

    return corridor if not corridor.empty else grid.copy()


aco_route = []
aco_cost = float("inf")

if routing_algorithm in ["ACO", "Compare A* vs ACO"]:

    aco_grid = build_aco_corridor(
        navigation_grid,
        start_lat,
        start_lon,
        goal_lat,
        goal_lon,
        margin=5.0,
    )

    aco_nodes = set(
        (
            round(float(row.grid_lat), 1),
            round(float(row.grid_lon), 1),
        )
        for row in aco_grid.itertuples()
    )

    aco_start = nearest_grid_node(
        start_lat,
        start_lon,
        aco_nodes,
    )

    aco_goal = nearest_grid_node(
        goal_lat,
        goal_lon,
        aco_nodes,
    )

    if aco_start is not None and aco_goal is not None:

        aco_router = ACOGridRouter(
            grid=aco_grid,
            start=aco_start,
            goal=aco_goal,
            risk_weight=risk_weight,
            iceberg_weight=iceberg_weight,
            n_ants=aco_ants,
            n_iterations=aco_iterations,
            alpha=aco_alpha,
            beta=aco_beta,
            evaporation_rate=aco_evaporation,
            pheromone_deposit=100.0,
            max_steps=300,
            seed=42,
        )

        aco_route, aco_cost = aco_router.solve()

        if aco_route is None:
            aco_route = []

if routing_algorithm == "A*":
    route = astar_route
    active_algorithm = "A*"
elif routing_algorithm == "ACO":
    route = aco_route
    active_algorithm = "ACO"
else:
    route = astar_route
    active_algorithm = "A*"


# ============================================================
# ROUTE STATISTICS
# ============================================================

def calculate_route_statistics(route_data, grid):

    if not route_data:
        return {
            "distance": 0.0,
            "average_risk": 0.0,
            "max_risk": 0.0,
            "fuel": 0.0,
        }

    risk_lookup = {}

    for row in grid.itertuples():
        node = (
            round(float(row.grid_lat), 1),
            round(float(row.grid_lon), 1),
        )
        risk_lookup[node] = float(
            getattr(row, "navigation_risk", 0.0)
        )

    distance = 0.0
    risks = []

    for i in range(len(route_data) - 1):
        lat1, lon1 = route_data[i]
        lat2, lon2 = route_data[i + 1]

        distance += haversine_km(
            lat1, lon1, lat2, lon2
        )

        risks.append(
            risk_lookup.get(
                (round(lat1, 1), round(lon1, 1)),
                0.0,
            )
        )

    risks.append(
        risk_lookup.get(
            (round(route_data[-1][0], 1), round(route_data[-1][1], 1)),
            0.0,
        )
    )

    average_risk = float(np.mean(risks)) if risks else 0.0
    maximum_risk = float(np.max(risks)) if risks else 0.0
    fuel = distance * 2.5 * (1.0 + average_risk)

    return {
        "distance": distance,
        "average_risk": average_risk,
        "max_risk": maximum_risk,
        "fuel": fuel,
    }


astar_stats = calculate_route_statistics(
    astar_route, navigation_grid
)

aco_stats = calculate_route_statistics(
    aco_route, navigation_grid
)

route_distance = (
    astar_stats["distance"]
    if active_algorithm == "A*"
    else aco_stats["distance"]
)

average_route_risk = (
    astar_stats["average_risk"]
    if active_algorithm == "A*"
    else aco_stats["average_risk"]
)

max_route_risk = (
    astar_stats["max_risk"]
    if active_algorithm == "A*"
    else aco_stats["max_risk"]
)

estimated_fuel = (
    astar_stats["fuel"]
    if active_algorithm == "A*"
    else aco_stats["fuel"]
)


# ============================================================
# ICEBERG CURRENT LOCATIONS
# ============================================================

if "hours_ahead" in icebergs.columns:

    current_icebergs = (
        icebergs.sort_values(
            "hours_ahead"
        )
        .groupby(
            "iceberg_id",
            as_index=False,
        )
        .first()
    )

else:

    current_icebergs = icebergs.copy()


# ============================================================
# CREATE MAIN ANTARCTIC MAP
# ============================================================

fig = go.Figure()


# ============================================================
# SEA ICE LAYER
# ============================================================

if show_sea_ice:

    ice_plot = ice[
        ice["sea_ice_concentration"] > 2
    ].copy()

    # Downsample for performance.
    if len(ice_plot) > 12000:

        ice_plot = ice_plot.sample(
            12000,
            random_state=42,
        )

    fig.add_trace(
        go.Scattergeo(
            lat=ice_plot["latitude"],
            lon=ice_plot["longitude"],
            mode="markers",
            name="Sea Ice",
            legendgroup="sea_ice",
            marker=dict(
                size=3.5,
                color=ice_plot[
                    "sea_ice_concentration"
                ],
                colorscale="Blues",
                cmin=0,
                cmax=100,
                opacity=0.50,
                colorbar=dict(
                    title="Sea Ice %",
                    x=1.02,
                ),
            ),
            customdata=np.column_stack(
                [
                    ice_plot[
                        "sea_ice_concentration"
                    ]
                ]
            ),
            hovertemplate=(
                "<b>Sea Ice</b><br>"
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°<br>"
                "Concentration: %{customdata[0]:.1f}%"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# ENVIRONMENTAL RISK LAYER
# ============================================================

if show_risk:

    risk_plot = navigation_grid[
        navigation_grid[
            "navigation_risk"
        ] > 0.05
    ].copy()

    if len(risk_plot) > 10000:

        risk_plot = risk_plot.sample(
            10000,
            random_state=42,
        )

    fig.add_trace(
        go.Scattergeo(
            lat=risk_plot["grid_lat"],
            lon=risk_plot["grid_lon"],
            mode="markers",
            name="Navigation Risk",
            legendgroup="risk",
            marker=dict(
                size=6,
                color=risk_plot[
                    "navigation_risk"
                ],
                colorscale="RdYlGn_r",
                cmin=0,
                cmax=1,
                opacity=0.35,
                colorbar=dict(
                    title="Risk",
                    x=1.10,
                ),
            ),
            customdata=np.column_stack(
                [
                    risk_plot[
                        "navigation_risk"
                    ],
                    risk_plot[
                        "iceberg_risk"
                    ],
                    risk_plot[
                        "risk_score"
                    ],
                ]
            ),
            hovertemplate=(
                "<b>Navigation Risk</b><br>"
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°<br>"
                "Combined Risk: %{customdata[0]:.2f}<br>"
                "Iceberg Risk: %{customdata[1]:.2f}<br>"
                "Environmental Risk: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# ICEBERG TRAJECTORIES
# ============================================================

if show_trajectories:

    trajectory_data = icebergs.copy()

    if "hours_ahead" in trajectory_data.columns:

        trajectory_data = trajectory_data[
            trajectory_data["hours_ahead"]
            <= trajectory_hours
        ]

    iceberg_ids = (
        trajectory_data[
            "iceberg_id"
        ]
        .astype(str)
        .unique()
    )

    # Plotly qualitative sequence
    trajectory_colors = [
        "#00B4D8",
        "#90BE6D",
        "#F9C74F",
        "#F9844A",
        "#577590",
        "#C77DFF",
        "#4CC9F0",
        "#F72585",
        "#43AA8B",
        "#FFB703",
    ]

    for index, iceberg_id in enumerate(
        iceberg_ids
    ):

        track = trajectory_data[
            trajectory_data[
                "iceberg_id"
            ].astype(str)
            == iceberg_id
        ].sort_values(
            "hours_ahead"
            if "hours_ahead" in trajectory_data.columns
            else "latitude"
        )

        color = trajectory_colors[
            index
            % len(trajectory_colors)
        ]

        if len(track) < 2:
            continue

        if "hours_ahead" in track.columns:

            custom = np.column_stack(
                [
                    track[
                        "hours_ahead"
                    ].to_numpy()
                ]
            )

            hover = (
                "<b>Iceberg "
                + iceberg_id
                + "</b><br>"
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°<br>"
                "Forecast: %{customdata[0]} h"
                "<extra></extra>"
            )

        else:

            custom = None

            hover = (
                "<b>Iceberg "
                + iceberg_id
                + "</b><br>"
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°"
                "<extra></extra>"
            )

        fig.add_trace(
            go.Scattergeo(
                lat=track["latitude"],
                lon=track["longitude"],
                mode="lines",
                name=f"Iceberg {iceberg_id}",
                legendgroup=f"iceberg_{iceberg_id}",
                showlegend=False,
                line=dict(
                    color=color,
                    width=2.5,
                ),
                customdata=custom,
                hovertemplate=hover,
            )
        )

    # Dummy legend entry
    fig.add_trace(
        go.Scattergeo(
            lat=[None],
            lon=[None],
            mode="lines",
            name="Iceberg Trajectories",
            line=dict(
                color="#00B4D8",
                width=3,
            ),
        )
    )


# ============================================================
# CURRENT ICEBERG POSITIONS
# ============================================================

if show_icebergs and not current_icebergs.empty:

    custom_columns = []

    if "iceberg_id" in current_icebergs.columns:
        custom_columns.append(
            current_icebergs[
                "iceberg_id"
            ].astype(str)
        )

    if "hours_ahead" in current_icebergs.columns:
        custom_columns.append(
            current_icebergs[
                "hours_ahead"
            ]
        )

    if custom_columns:

        custom = np.column_stack(
            [
                column.to_numpy()
                for column in custom_columns
            ]
        )

    else:

        custom = None

    fig.add_trace(
        go.Scattergeo(
            lat=current_icebergs[
                "latitude"
            ],
            lon=current_icebergs[
                "longitude"
            ],
            mode="markers",
            name="Current Icebergs",
            marker=dict(
                size=11,
                symbol="diamond",
                color="#FF4D4D",
                line=dict(
                    width=1.5,
                    color="white",
                ),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>🧊 Iceberg</b><br>"
                "ID: %{customdata[0]}<br>"
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°<br>"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# OPTIMIZED ROUTE
# ============================================================

if show_route and route:

    route_lat = [point[0] for point in route]
    route_lon = [point[1] for point in route]

    route_color = "#00FF66" if active_algorithm == "A*" else "#FFB703"

    fig.add_trace(
        go.Scattergeo(
            lat=route_lat,
            lon=route_lon,
            mode="lines",
            name=f"{active_algorithm} Optimized Route",
            line=dict(
                color=route_color,
                width=5,
            ),
            hovertemplate=(
                f"<b>{active_algorithm} Optimized Route</b><br>"
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# A* vs ACO COMPARISON ROUTE
# ============================================================

if routing_algorithm == "Compare A* vs ACO" and aco_route:

    aco_lat = [point[0] for point in aco_route]
    aco_lon = [point[1] for point in aco_route]

    fig.add_trace(
        go.Scattergeo(
            lat=aco_lat,
            lon=aco_lon,
            mode="lines",
            name="ACO Route",
            line=dict(
                color="#FFB703",
                width=4,
                dash="dash",
            ),
            hovertemplate=(
                "<b>ACO Route</b><br>"
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# VESSEL START
# ============================================================

fig.add_trace(
    go.Scattergeo(
        lat=[start_lat],
        lon=[start_lon],
        mode="markers",
        name="Vessel",
        marker=dict(
            size=15,
            symbol="circle",
            color="white",
            line=dict(
                color="#00FF66",
                width=3,
            ),
        ),
        hovertemplate=(
            "<b>🚢 Vessel</b><br>"
            f"{vessel_name}<br>"
            "Latitude: %{lat:.2f}°<br>"
            "Longitude: %{lon:.2f}°"
            "<extra></extra>"
        ),
    )
)


# ============================================================
# DESTINATION
# ============================================================

fig.add_trace(
    go.Scattergeo(
        lat=[goal_lat],
        lon=[goal_lon],
        mode="markers",
        name="Destination",
        marker=dict(
            size=18,
            symbol="star",
            color="#FFD166",
            line=dict(
                color="white",
                width=1,
            ),
        ),
        hovertemplate=(
            "<b>⭐ Destination</b><br>"
            "Latitude: %{lat:.2f}°<br>"
            "Longitude: %{lon:.2f}°"
            "<extra></extra>"
        ),
    )
)


# ============================================================
# ANTARCTIC GEOGRAPHIC STYLE
# ============================================================

fig.update_geos(
    projection_type="stereographic",
    projection_rotation=dict(
        lat=-90,
        lon=0,
    ),
    center=dict(
        lat=-90,
        lon=0,
    ),
    projection_scale=1.35,

    showland=True,
    landcolor="#D9E1E5",

    showocean=True,
    oceancolor="#061A2B",

    showlakes=False,

    showcountries=True,
    countrycolor="#718999",

    showcoastlines=True,
    coastlinecolor="#FFFFFF",
    coastlinewidth=1.2,

    showframe=True,
    framecolor="#4E7187",

    lataxis_showgrid=True,
    lataxis_gridcolor="#38576A",
    lataxis_gridwidth=0.6,
    lataxis_dtick=10,

    lonaxis_showgrid=True,
    lonaxis_gridcolor="#38576A",
    lonaxis_gridwidth=0.6,
    lonaxis_dtick=30,
)


# ============================================================
# MAP LAYOUT
# ============================================================

fig.update_layout(

    title=dict(
        text=(
            "POLARIS — Antarctic Navigation & "
            "Iceberg Risk Map"
        ),
        font=dict(
            size=22,
            color="white",
        ),
        x=0.02,
        xanchor="left",
    ),

    height=800,

    paper_bgcolor="#07131F",

    plot_bgcolor="#07131F",

    margin=dict(
        l=10,
        r=120,
        t=70,
        b=10,
    ),

    legend=dict(
        bgcolor="rgba(7,19,31,0.85)",
        bordercolor="#31566D",
        borderwidth=1,
        font=dict(
            color="white",
        ),
        orientation="v",
        y=0.98,
        x=0.01,
    ),

)


# ============================================================
# DISPLAY MAP
# ============================================================

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
    },
)


# ============================================================
# KPI SECTION
# ============================================================

st.markdown("### Mission Intelligence")

col1, col2, col3, col4, col5 = st.columns(5)


# Average sea ice
avg_ice = float(
    ice[
        "sea_ice_concentration"
    ].mean()
)


# High-risk percentage
high_risk_percentage = (
    (
        risk[
            "risk_score"
        ] >= 0.60
    ).mean()
    * 100
)


# Number of icebergs
tracked_icebergs = (
    current_icebergs[
        "iceberg_id"
    ].nunique()
    if "iceberg_id"
    in current_icebergs.columns
    else len(current_icebergs)
)


# ------------------------------------------------------------
# KPI 1
# ------------------------------------------------------------

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Average Sea Ice
            </div>
            <div class="metric-value">
                {avg_ice:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# KPI 2
# ------------------------------------------------------------

with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Avg Navigation Risk
            </div>
            <div class="metric-value">
                {average_route_risk:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# KPI 3
# ------------------------------------------------------------

with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                High Risk Area
            </div>
            <div class="metric-value">
                {high_risk_percentage:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# KPI 4
# ------------------------------------------------------------

with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Tracked Icebergs
            </div>
            <div class="metric-value">
                {tracked_icebergs}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# KPI 5
# ------------------------------------------------------------

with col5:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Route Distance
            </div>
            <div class="metric-value">
                {route_distance:.0f} km
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ROUTE ANALYSIS
# ============================================================

st.markdown("---")

st.markdown("### 🧭 AI-Assisted Navigation")


if route:

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Optimized Distance",
            f"{route_distance:.1f} km",
        )

    with r2:
        st.metric(
            "Average Route Risk",
            f"{average_route_risk:.3f}",
        )

    with r3:
        st.metric(
            "Maximum Route Risk",
            f"{max_route_risk:.3f}",
        )

    with r4:
        st.metric(
            "Estimated Fuel",
            f"{estimated_fuel:,.0f} units",
        )

    st.success(
        f"POLARIS generated a risk-aware navigation route using {active_algorithm}."
    )

    if routing_algorithm == "Compare A* vs ACO":

        st.markdown("#### 🧠 A* vs ACO")

        comparison = pd.DataFrame([
            {
                "Algorithm": "A*",
                "Distance (km)": round(astar_stats["distance"], 2),
                "Average Risk": round(astar_stats["average_risk"], 3),
                "Maximum Risk": round(astar_stats["max_risk"], 3),
                "Estimated Fuel": round(astar_stats["fuel"], 1),
                "Route Points": len(astar_route),
            },
            {
                "Algorithm": "ACO",
                "Distance (km)": round(aco_stats["distance"], 2),
                "Average Risk": round(aco_stats["average_risk"], 3),
                "Maximum Risk": round(aco_stats["max_risk"], 3),
                "Estimated Fuel": round(aco_stats["fuel"], 1),
                "Route Points": len(aco_route),
            },
        ])

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )

else:

    st.warning(
        "No valid route could be generated for "
        "the selected start and destination."
    )


# ============================================================
# ROUTE EXPLANATION
# ============================================================

with st.expander(
    "🔬 How POLARIS generates the route"
):

    st.markdown(
        """
        **1. Environmental intelligence**

        POLARIS combines:

        - NSIDC sea-ice concentration
        - ERA5 wind
        - ERA5 temperature
        - ERA5 significant wave height

        **2. Iceberg intelligence**

        The predicted iceberg trajectories are converted
        into a spatial iceberg-risk field.

        **3. Risk grid**

        Antarctica is converted into a navigation grid.
        Each grid cell receives a combined risk value.

        **4. A* optimization**

        A* provides a fast deterministic baseline route.
        It increases movement cost when the vessel enters
        cells with higher environmental or iceberg risk.

        **5. Ant Colony Optimization**

        ACO uses multiple artificial ants to explore
        candidate routes. Good routes receive pheromone
        reinforcement while pheromone evaporation prevents
        permanent dependence on older search paths.

        **6. Route comparison**

        POLARIS can compare A* and ACO using distance,
        average navigation risk, maximum navigation risk
        and a simplified fuel estimate.

        The fuel figure shown here is a simplified
        prototype estimate and should not be interpreted
        as a naval-grade propulsion calculation.
        """
    )


# ============================================================
# ICEBERG INTELLIGENCE
# ============================================================

with st.expander(
    "🧊 Iceberg Intelligence"
):

    if not current_icebergs.empty:

        iceberg_table = current_icebergs.copy()

        display_columns = [
            column
            for column in [
                "iceberg_id",
                "latitude",
                "longitude",
                "hours_ahead",
            ]
            if column in iceberg_table.columns
        ]

        iceberg_table = iceberg_table[
            display_columns
        ]

        st.dataframe(
            iceberg_table,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No iceberg prediction data available."
        )


# ============================================================
# ENVIRONMENTAL DATA
# ============================================================

with st.expander(
    "🌎 Environmental Intelligence"
):

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Mean Wind",
            f"{risk['wind_speed'].mean():.2f} m/s"
            if "wind_speed" in risk.columns
            else "N/A",
        )

    with c2:

        st.metric(
            "Mean Temperature",
            f"{risk['temperature'].mean():.1f} °C"
            if "temperature" in risk.columns
            else "N/A",
        )

    with c3:

        st.metric(
            "Mean Wave Height",
            f"{risk['wave_height'].mean():.2f} m"
            if "wave_height" in risk.columns
            else "N/A",
        )


# ============================================================
# DATA SOURCE STATUS
# ============================================================

st.markdown("---")

st.caption(
    """
    POLARIS prototype data:
    NSIDC passive microwave sea-ice concentration +
    ERA5 atmospheric/wave conditions +
    AI-generated iceberg trajectories.
    """
)

st.caption(
    "⚠️ Current prototype uses the locally processed dataset. "
    "The map is interactive, but the underlying data is not "
    "yet a live real-time feed."
)