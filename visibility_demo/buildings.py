"""Building outlines -> façade points.

Exposure is attributed to points on the outlines, so parts of one wall can differ. Points are at most `spacing` metres
apart: short walls still get points, long walls are sampled evenly.
"""
import json

import numpy as np


def load_outlines(path, proj):
    """Outer rings of all polygons in a GeoJSON file, as (n, 2) arrays in metres."""
    with open(path, encoding="utf-8") as f:
        features = json.load(f)["features"]
    rings = []
    for feat in features:
        geom = feat["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polygons:
            rings.append(np.column_stack(proj.to_xy(*np.asarray(poly[0], float).T)))
    return rings


def densify(line, spacing=10.0):
    """Points along a polyline, at most `spacing` apart; the last vertex is not repeated."""
    parts = []
    for a, b in zip(line[:-1], line[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / spacing)))
        parts.append(a + np.arange(n)[:, None] / n * (b - a))
    return np.vstack(parts)


def facade_points(rings, spacing=10.0):
    """Densified, deduplicated outline points and the index of the building each came from."""
    pts = [densify(r, spacing) for r in rings]
    owner = np.repeat(np.arange(len(rings)), [len(p) for p in pts])
    pts = np.vstack(pts)
    _, first = np.unique(pts.round(2), axis=0, return_index=True)   # shared walls count once
    first.sort()
    return pts[first], owner[first]
