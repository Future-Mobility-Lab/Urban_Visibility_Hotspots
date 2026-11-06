"""Synthetic trips on the OpenStreetMap road network (the paper's trajectories cannot be shared).

    graph    road vertices as nodes, free-flow speed per road class; one-way rules ignored
    route    fastest path between random nodes >= 1.2 km apart (traffic gathers on arterials)
    motion   random fraction of free-flow speed; random stops at major junctions (signals)
    output   positions every 5 s with the columns of a connected-vehicle table
"""
import json

import networkx as nx
import numpy as np
import pandas as pd

from .geo import bearing_deg

FREE_FLOW_MPS = {"motorway": 22.0, "trunk": 18.0, "primary": 14.0, "secondary": 12.5, "tertiary": 11.0,
                 "unclassified": 9.0, "residential": 8.0, "living_street": 5.0}
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary"}


def road_class(highway):
    return highway.removesuffix("_link")


def road_graph(path, proj):
    """Largest connected road graph; edges: length, speed, time, cls; nodes: major (signal-like junction)."""
    with open(path, encoding="utf-8") as f:
        features = json.load(f)["features"]
    g = nx.Graph()
    for feat in features:
        cls = road_class(feat["properties"]["highway"])
        if cls not in FREE_FLOW_MPS:
            continue
        x, y = proj.to_xy(*np.asarray(feat["geometry"]["coordinates"], float).T)
        nodes = list(zip(x.round(1), y.round(1)))
        for u, v in zip(nodes[:-1], nodes[1:]):
            if u != v:
                length = float(np.hypot(v[0] - u[0], v[1] - u[1]))
                speed = FREE_FLOW_MPS[cls]
                g.add_edge(u, v, length=length, speed=speed, time=length / speed, cls=cls)
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    for n in g:
        g.nodes[n]["major"] = sum(g.edges[n, m]["cls"] in MAJOR for m in g[n]) >= 3
    return g


def drive(g, route, rng, stop_prob=0.3, max_stop_s=40.0):
    """Arrays t (s), x, y, speed (m/s) along a node route, at about 1 s resolution."""
    t, x, y, v = [0.0], [route[0][0]], [route[0][1]], [0.0]

    def step(dt, px, py, speed):
        t.append(t[-1] + dt); x.append(px); y.append(py); v.append(speed)

    for u, w in zip(route[:-1], route[1:]):
        e = g.edges[u, w]
        speed = e["speed"] * rng.uniform(0.55, 0.95)
        n = max(1, int(np.ceil(e["length"] / speed)))
        for f in np.arange(1, n + 1) / n:
            step(e["length"] / n / speed, u[0] + f * (w[0] - u[0]), u[1] + f * (w[1] - u[1]), speed)
        if g.nodes[w]["major"] and w != route[-1] and rng.random() < stop_prob:
            step(1e-3, x[-1], y[-1], 0.0)                               # arrive
            step(rng.uniform(5.0, max_stop_s), x[-1], y[-1], 0.0)       # wait
    return map(np.array, (t, x, y, v))


def sample(t, x, y, v, interval=5.0):
    """Positions every `interval` s; bearing = direction of travel, held while stopped."""
    ts = np.arange(0.0, t[-1] + 1e-9, interval)
    xs, ys, vs = (np.interp(ts, t, a) for a in (x, y, v))
    dx, dy = np.diff(xs, append=xs[-1]), np.diff(ys, append=ys[-1])
    b = np.where(np.hypot(dx, dy) > 0.5, bearing_deg(dx, dy), np.nan)
    return ts, xs, ys, pd.Series(b).ffill().bfill().to_numpy(), vs


def generate_trips(g, proj, n_trips=1500, seed=7, min_od_m=1200.0, start="2024-01-01", days=7):
    rng = np.random.default_rng(seed)
    nodes = list(g)
    xy = np.array(nodes)
    t0 = pd.Timestamp(start, tz="UTC")
    trips = []
    while len(trips) < n_trips:
        a, b = rng.integers(len(nodes), size=2)
        if np.hypot(*(xy[a] - xy[b])) < min_od_m:
            continue
        ts, xs, ys, bs, vs = sample(*drive(g, nx.shortest_path(g, nodes[a], nodes[b], weight="time"), rng))
        lon, lat = proj.to_lonlat(xs, ys)
        begin = t0 + pd.Timedelta(seconds=float(rng.uniform(0, days * 86400)))
        trips.append(pd.DataFrame({"trip_id": f"S{len(trips):05d}", "timestamp": begin + pd.to_timedelta(ts, unit="s"),
                                   "lat": lat.round(7), "lon": lon.round(7), "bearing": bs.round(2),
                                   "speed_mps": vs.round(2)}))
    return pd.concat(trips, ignore_index=True)
