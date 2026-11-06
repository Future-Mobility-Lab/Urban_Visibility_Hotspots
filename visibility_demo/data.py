"""Loaders for data/. Everything is converted to metres around a reference point in Waterloo, Sydney."""
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .buildings import facade_points, load_outlines
from .exposure import build_index
from .geo import LocalProjection

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGURES, OUTPUTS = ROOT / "data", ROOT / "figures", ROOT / "outputs"
PROJ = LocalProjection(-33.902891, 151.207726)


def load_trips():
    """Synthetic trips, one row per 5-s position, with x, y (m) and an integer trip_code."""
    df = pd.read_csv(DATA / "trips_synthetic.csv.gz", parse_dates=["timestamp"])
    df["x"], df["y"] = PROJ.to_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["trip_code"] = pd.factorize(df["trip_id"])[0]
    return df


def load_roads():
    """[(highway class, (n, 2) array in metres), ...]"""
    with open(DATA / "roads_waterloo.geojson", encoding="utf-8") as f:
        features = json.load(f)["features"]
    return [(ft["properties"]["highway"], np.column_stack(PROJ.to_xy(*np.asarray(ft["geometry"]["coordinates"]).T)))
            for ft in features]


@dataclass
class Scene:
    """All demo inputs, loaded once: building rings, façade points, trips, roads, and the index (built on first use)."""
    rings: list
    points: np.ndarray
    owner: np.ndarray
    trips: pd.DataFrame
    roads: list = field(default_factory=load_roads)

    @property
    def xyb(self):
        """x, y, bearing of every position."""
        return (self.trips[c].to_numpy(float) for c in ("x", "y", "bearing"))

    @cached_property
    def tree(self):
        return build_index(self.points)

    def bounds(self, pad=60):
        x, y = self.trips["x"], self.trips["y"]
        return x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad


def load_scene(spacing=10.0):
    rings = load_outlines(DATA / "buildings_waterloo.geojson", PROJ)
    points, owner = facade_points(rings, spacing)
    return Scene(rings, points, owner, load_trips())


def output(name):
    """Path in outputs/, creating the folder."""
    OUTPUTS.mkdir(exist_ok=True)
    return OUTPUTS / name
