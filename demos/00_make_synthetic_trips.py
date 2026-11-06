"""Demo 00: regenerate data/trips_synthetic.csv.gz (already included).

    python demos/00_make_synthetic_trips.py [--trips 1500] [--seed 7]
"""
import argparse

import _path  # noqa: F401

from visibility_demo.data import DATA, PROJ
from visibility_demo.synthetic import generate_trips, road_graph

ap = argparse.ArgumentParser()
ap.add_argument("--trips", type=int, default=1500)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

g = road_graph(DATA / "roads_waterloo.geojson", PROJ)
print(f"road graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges, "
      f"{sum(d['major'] for _, d in g.nodes(data=True))} major junctions")
df = generate_trips(g, PROJ, n_trips=args.trips, seed=args.seed)
df.to_csv(DATA / "trips_synthetic.csv.gz", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
print(f"{df.trip_id.nunique():,} trips, {len(df):,} positions, {(df.speed_mps == 0).mean():.1%} stopped")
