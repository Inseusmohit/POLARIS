"""POLARIS - destination-aware Ant Colony Optimization router.

This router is designed for the POLARIS Antarctic navigation grid and is
compatible with the existing dashboard interface:

    route, cost = ACOGridRouter(...).solve()

Design goals:
- Minimize a navigation cost that includes distance/fuel, environmental risk,
  iceberg risk, detour length, and unnecessary turns.
- Keep ants strongly directed toward the destination while still allowing
  controlled detours around risky cells.
- Use edge pheromone rather than node-only pheromone so a specific movement
  direction can be reinforced.
- Prevent loops and extreme wandering with a progress gate and route-length
  limit.
- Keep the public constructor compatible with the current POLARIS app.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Node = Tuple[float, float]
Edge = Tuple[Node, Node]


class ACOGridRouter:
    """ACO path planner for a 0.5-degree Antarctic navigation grid."""

    def __init__(
        self,
        grid: pd.DataFrame,
        start: Node,
        goal: Node,
        risk_weight: float = 15.0,
        iceberg_weight: float = 12.0,
        n_ants: int = 25,
        n_iterations: int = 30,
        alpha: float = 1.0,
        beta: float = 4.0,
        evaporation_rate: float = 0.35,
        pheromone_deposit: float = 100.0,
        max_steps: int = 300,
        seed: int = 42,
        # Additional controls.  These have sensible defaults so the current
        # dashboard does not need to pass them.
        fuel_weight: float = 1.0,
        detour_weight: float = 3.0,
        turn_weight: float = 0.10,
        progress_weight: float = 5.0,
        max_detour_ratio: float = 1.65,
        corridor_margin_deg: float = 5.0,
    ):
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))

        self.risk_weight = float(risk_weight)
        self.iceberg_weight = float(iceberg_weight)
        self.n_ants = max(1, int(n_ants))
        self.n_iterations = max(1, int(n_iterations))
        self.alpha = max(0.0, float(alpha))
        self.beta = max(0.1, float(beta))
        self.evaporation_rate = min(0.95, max(0.01, float(evaporation_rate)))
        self.pheromone_deposit = max(1e-6, float(pheromone_deposit))
        self.max_steps_requested = max(10, int(max_steps))
        self.fuel_weight = max(0.0, float(fuel_weight))
        self.detour_weight = max(0.0, float(detour_weight))
        self.turn_weight = max(0.0, float(turn_weight))
        self.progress_weight = max(0.0, float(progress_weight))
        self.max_detour_ratio = max(1.05, float(max_detour_ratio))
        self.corridor_margin_deg = max(0.5, float(corridor_margin_deg))
        self.rng = random.Random(seed)

        self.grid = self._prepare_grid(grid)
        self.nodes = set(self.grid.keys())
        if not self.nodes:
            raise ValueError("ACO navigation grid is empty.")

        self.start_node = self._nearest_node(self.start)
        self.goal_node = self._nearest_node(self.goal)

        self.neighbors = {
            node: self._get_neighbors(node)
            for node in self.nodes
        }

        # Edge pheromone is important: pheromone on (A -> B) should not make
        # the reverse movement (B -> A) equally attractive.
        self.pheromone: Dict[Edge, float] = {}
        for node in self.nodes:
            for nxt in self.neighbors[node]:
                self.pheromone[(node, nxt)] = self._initial_pheromone(node, nxt)

        self.direct_distance = self._haversine_km(
            self.start_node,
            self.goal_node,
        )

        # A 0.5-degree grid has at least ~55 km cardinal spacing near the
        # equator, less at Antarctic latitudes. Use the shortest direct-grid
        # edge as a reference for the route-length limit.
        self.min_edge_km = self._estimate_min_edge()
        direct_steps = max(1, int(math.ceil(self.direct_distance / max(self.min_edge_km, 1e-6))))
        self.max_steps = min(
            self.max_steps_requested,
            max(20, int(direct_steps * self.max_detour_ratio) + 8),
        )

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine_km(a: Node, b: Node) -> float:
        r = 6371.0
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat = lat2 - lat1
        dlon = math.radians(
            ((b[1] - a[1] + 180.0) % 360.0) - 180.0
        )
        x = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2.0) ** 2
        )
        x = min(1.0, max(0.0, x))
        return 2.0 * r * math.asin(math.sqrt(x))

    @staticmethod
    def _normalize_lon(lon: float) -> float:
        value = ((float(lon) + 180.0) % 360.0) - 180.0
        # Keep +180 where possible for consistency with source grids.
        if abs(value + 180.0) < 1e-9:
            return 180.0
        return value

    def _nearest_node(self, point: Node) -> Node:
        return min(
            self.nodes,
            key=lambda n: self._haversine_km(n, point),
        )

    # ------------------------------------------------------------------
    # Grid preparation
    # ------------------------------------------------------------------

    def _prepare_grid(self, grid: pd.DataFrame) -> Dict[Node, Dict[str, float]]:
        required = {"grid_lat", "grid_lon"}
        missing = required - set(grid.columns)
        if missing:
            raise ValueError(
                f"ACO grid is missing columns: {sorted(missing)}"
            )

        df = grid.copy()

        # ACO works inside a mission corridor.  The dashboard already creates
        # a broad corridor; this second filter keeps the router itself safe if
        # it is called directly.
        min_lat = min(self.start[0], self.goal[0]) - self.corridor_margin_deg
        max_lat = max(self.start[0], self.goal[0]) + self.corridor_margin_deg

        lon_diff = abs(self.goal[1] - self.start[1])
        if lon_diff <= 180.0:
            min_lon = min(self.start[1], self.goal[1]) - self.corridor_margin_deg
            max_lon = max(self.start[1], self.goal[1]) + self.corridor_margin_deg
            df = df[
                df["grid_lon"].between(min_lon, max_lon)
            ]

        df = df[
            df["grid_lat"].between(min_lat, max_lat)
        ].copy()

        if df.empty:
            df = grid.copy()

        records: Dict[Node, Dict[str, float]] = {}
        for row in df.itertuples(index=False):
            node = (
                round(float(row.grid_lat), 1),
                round(float(row.grid_lon), 1),
            )

            env = float(
                getattr(
                    row,
                    "risk_score",
                    getattr(row, "navigation_risk", 0.0),
                )
            )
            iceberg = float(
                getattr(row, "iceberg_risk", 0.0)
            )
            navigation = float(
                getattr(
                    row,
                    "navigation_risk",
                    0.75 * env + 0.25 * iceberg,
                )
            )

            records[node] = {
                "environmental_risk": float(np.clip(env, 0.0, 1.0)),
                "iceberg_risk": float(np.clip(iceberg, 0.0, 1.0)),
                "navigation_risk": float(np.clip(navigation, 0.0, 1.0)),
            }

        return records

    def _get_neighbors(self, node: Node) -> List[Node]:
        lat, lon = node
        directions = (
            (-0.5, 0.0),
            (0.5, 0.0),
            (0.0, -0.5),
            (0.0, 0.5),
            (-0.5, -0.5),
            (-0.5, 0.5),
            (0.5, -0.5),
            (0.5, 0.5),
        )

        result = []
        for dlat, dlon in directions:
            candidate = (
                round(lat + dlat, 1),
                round(self._normalize_lon(lon + dlon), 1),
            )
            if candidate in self.nodes:
                result.append(candidate)
        return result

    def _estimate_min_edge(self) -> float:
        values = []
        for node in self.nodes:
            for nxt in self.neighbors[node]:
                values.append(self._haversine_km(node, nxt))
        return min(values) if values else 1.0

    # ------------------------------------------------------------------
    # Cost / heuristic
    # ------------------------------------------------------------------

    def movement_cost(self, a: Node, b: Node) -> float:
        """Cost of physically moving from a to b."""
        distance = self._haversine_km(a, b)
        cell = self.grid[b]

        # Distance is the base fuel proxy.  Risk increases the effective
        # operational cost of entering a cell.
        risk_multiplier = (
            1.0
            + self.risk_weight * cell["environmental_risk"] / 15.0
            + self.iceberg_weight * cell["iceberg_risk"] / 12.0
        )

        return distance * self.fuel_weight * risk_multiplier

    def _progress(self, node: Node) -> float:
        """1 at the goal, 0 at the start, based on direct-line progress."""
        remaining = self._haversine_km(node, self.goal_node)
        if self.direct_distance <= 1e-9:
            return 1.0
        return float(
            np.clip(
                1.0 - remaining / self.direct_distance,
                0.0,
                1.0,
            )
        )

    def _turn_penalty(self, previous: Optional[Node], current: Node, nxt: Node) -> float:
        """Penalize sharp direction changes to discourage zig-zag routes."""
        if previous is None:
            return 0.0

        v1 = (
            current[0] - previous[0],
            current[1] - previous[1],
        )
        v2 = (
            nxt[0] - current[0],
            nxt[1] - current[1],
        )

        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 == 0 or n2 == 0:
            return 0.0

        cosine = np.clip(
            (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2),
            -1.0,
            1.0,
        )
        # 0 for straight, 1 for 90 degrees, 2 for a reversal.
        return float(1.0 - cosine)

    def _candidate_allowed(self, current: Node, nxt: Node) -> bool:
        """Reject moves that create extreme backward detours."""
        current_remaining = self._haversine_km(current, self.goal_node)
        next_remaining = self._haversine_km(nxt, self.goal_node)

        # Normal moves should make progress. A small backward allowance lets
        # ants go around an isolated high-risk cell or obstacle.
        step = self._haversine_km(current, nxt)
        backward_allowance = max(12.0, 0.35 * step)

        return next_remaining <= current_remaining + backward_allowance

    def _heuristic(self, current: Node, nxt: Node) -> float:
        """Destination-aware desirability of an edge."""
        move_cost = self.movement_cost(current, nxt)
        remaining = self._haversine_km(nxt, self.goal_node)
        current_remaining = self._haversine_km(current, self.goal_node)

        step = max(self._haversine_km(current, nxt), 1.0)
        progress = current_remaining - remaining

        # Positive progress is rewarded; backward motion is strongly penalized.
        progress_ratio = progress / step
        progress_factor = math.exp(
            np.clip(self.progress_weight * progress_ratio, -8.0, 8.0)
        )

        # Future distance matters, but is kept bounded so risk avoidance can
        # still justify a reasonable detour.
        goal_factor = 1.0 / (
            1.0
            + move_cost
            + 0.08 * remaining
        )

        return max(1e-12, progress_factor * goal_factor)

    def route_cost(self, route: Sequence[Node]) -> float:
        if not route or len(route) < 2:
            return float("inf")

        total = 0.0
        previous = None
        for a, b in zip(route, route[1:]):
            total += self.movement_cost(a, b)
            total += self.turn_weight * self._turn_penalty(previous, a, b) * self._haversine_km(a, b)
            previous = a

        # Explicit detour penalty makes fuel efficiency part of the ACO
        # objective rather than merely a post-hoc dashboard metric.
        actual_distance = sum(
            self._haversine_km(a, b)
            for a, b in zip(route, route[1:])
        )
        detour_ratio = actual_distance / max(self.direct_distance, 1.0)
        if detour_ratio > 1.0:
            total += (
                self.detour_weight
                * self.direct_distance
                * (detour_ratio - 1.0) ** 2
            )

        return total

    # ------------------------------------------------------------------
    # Pheromone operations
    # ------------------------------------------------------------------

    def _initial_pheromone(self, a: Node, b: Node) -> float:
        # Give edges that point toward the goal a mild initial advantage.
        current_remaining = self._haversine_km(a, self.goal)
        next_remaining = self._haversine_km(b, self.goal)
        progress = current_remaining - next_remaining
        if progress > 0:
            return 1.0 + min(2.0, progress / max(self._haversine_km(a, b), 1.0))
        return 0.5

    def evaporate(self) -> None:
        factor = 1.0 - self.evaporation_rate
        for edge in self.pheromone:
            self.pheromone[edge] = max(
                1e-5,
                self.pheromone[edge] * factor,
            )

    def deposit(self, routes: Sequence[Sequence[Node]]) -> None:
        if not routes:
            return

        # Deposit only the best routes from the iteration. This reduces noisy
        # pheromone reinforcement from wandering ants.
        ranked = sorted(
            routes,
            key=self.route_cost,
        )[: min(5, len(routes))]

        for route in ranked:
            cost = max(self.route_cost(route), 1.0)
            amount = self.pheromone_deposit / cost
            for a, b in zip(route, route[1:]):
                edge = (a, b)
                if edge in self.pheromone:
                    self.pheromone[edge] += amount

    # ------------------------------------------------------------------
    # Ant construction
    # ------------------------------------------------------------------

    def _construct_ant(self) -> Optional[List[Node]]:
        current = self.start_node
        route = [current]
        visited = {current}
        previous = None

        for _ in range(self.max_steps):
            if current == self.goal_node:
                return route

            candidates = [
                nxt
                for nxt in self.neighbors.get(current, [])
                if nxt not in visited and self._candidate_allowed(current, nxt)
            ]

            # If the progress gate blocks every move, allow the least-bad
            # unvisited neighbor rather than declaring the ant dead.
            if not candidates:
                candidates = [
                    nxt
                    for nxt in self.neighbors.get(current, [])
                    if nxt not in visited
                ]

            if not candidates:
                return None

            scores = []
            for nxt in candidates:
                tau = max(
                    self.pheromone.get((current, nxt), 0.5),
                    1e-9,
                ) ** self.alpha

                eta = self._heuristic(current, nxt) ** self.beta

                turn = self._turn_penalty(previous, current, nxt)
                turn_factor = math.exp(
                    -self.turn_weight * turn
                )

                # A small explicit preference for getting closer to the goal.
                current_remaining = self._haversine_km(current, self.goal_node)
                next_remaining = self._haversine_km(nxt, self.goal_node)
                goal_progress = max(
                    -1.0,
                    min(
                        1.0,
                        (current_remaining - next_remaining)
                        / max(self._haversine_km(current, nxt), 1.0),
                    ),
                )
                goal_factor = math.exp(
                    np.clip(2.5 * goal_progress, -6.0, 6.0)
                )

                score = tau * eta * turn_factor * goal_factor
                scores.append(score)

            scores = np.asarray(scores, dtype=float)
            if not np.isfinite(scores).all() or scores.sum() <= 0:
                chosen = min(
                    candidates,
                    key=lambda n: self._haversine_km(n, self.goal_node),
                )
            else:
                probabilities = scores / scores.sum()
                chosen = self.rng_choice(candidates, probabilities)

            route.append(chosen)
            visited.add(chosen)
            previous, current = current, chosen

        return None

    def rng_choice(self, candidates: Sequence[Node], probabilities: np.ndarray) -> Node:
        index = self.rng.choices(
            range(len(candidates)),
            weights=probabilities.tolist(),
            k=1,
        )[0]
        return candidates[index]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed_pheromone(self, route: Sequence[Node], strength: float = 3.0) -> None:
        """Seed pheromone along a known route, e.g. an A* baseline."""
        for a, b in zip(route, route[1:]):
            edge = (a, b)
            if edge in self.pheromone:
                self.pheromone[edge] += float(strength)

    def update_risk(self, grid: pd.DataFrame, *, extra_evaporation: float = 0.20) -> None:
        """Refresh risk data while retaining only partially useful pheromone."""
        old_pheromone = self.pheromone.copy()
        self.grid = self._prepare_grid(grid)
        self.nodes = set(self.grid.keys())
        self.start_node = self._nearest_node(self.start)
        self.goal_node = self._nearest_node(self.goal)
        self.neighbors = {
            node: self._get_neighbors(node)
            for node in self.nodes
        }

        decay = max(0.0, min(1.0, 1.0 - extra_evaporation))
        self.pheromone = {}
        for node in self.nodes:
            for nxt in self.neighbors[node]:
                self.pheromone[(node, nxt)] = (
                    old_pheromone.get((node, nxt), self._initial_pheromone(node, nxt))
                    * decay
                )

        self.direct_distance = self._haversine_km(
            self.start_node,
            self.goal_node,
        )

    def solve(self):
        """Return (route, cost), matching the existing POLARIS dashboard."""
        best_route: Optional[List[Node]] = None
        best_cost = float("inf")

        for _ in range(self.n_iterations):
            successful_routes: List[List[Node]] = []

            for _ in range(self.n_ants):
                route = self._construct_ant()
                if route is None or route[-1] != self.goal_node:
                    continue

                cost = self.route_cost(route)
                if not math.isfinite(cost):
                    continue

                successful_routes.append(route)

                if cost < best_cost:
                    best_cost = cost
                    best_route = list(route)

            self.evaporate()
            self.deposit(successful_routes)

            # Strong elitist reinforcement helps the colony converge on the
            # best route discovered so far without allowing old pheromone to
            # dominate forever.
            if best_route is not None:
                self.deposit([best_route])

        return best_route or [], best_cost

    @staticmethod
    def route_to_coordinates(route: Sequence[Node]) -> List[Tuple[float, float]]:
        return [
            (float(lat), float(lon))
            for lat, lon in route
        ]

    def route_to_dataframe(self, route: Sequence[Node]) -> pd.DataFrame:
        rows = []
        for step, node in enumerate(route):
            cell = self.grid[node]
            rows.append(
                {
                    "step": step,
                    "latitude": node[0],
                    "longitude": node[1],
                    "environmental_risk": cell["environmental_risk"],
                    "iceberg_risk": cell["iceberg_risk"],
                    "navigation_risk": cell["navigation_risk"],
                }
            )
        return pd.DataFrame(rows)
