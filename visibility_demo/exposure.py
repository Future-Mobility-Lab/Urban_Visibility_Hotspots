"""Exposure counting: BallTree index, exhaustive baseline, traffic flow.

Exposure of a façade point = number of 5-s vehicle positions whose viewing area contains it.
The BallTree skips every ball of points that misses the query disc; results equal the exhaustive test.
"""
import numpy as np
from sklearn.neighbors import BallTree

from .shapes import TESTS, query_disc

CHUNK = 4096   # positions per batched query


def build_index(points, leaf_size=40):
    return BallTree(points, leaf_size=leaf_size)


def _chunks(n, size=CHUNK):
    return (slice(i, i + size) for i in range(0, n, size))


def _pairs(tree, qx, qy, r):
    """(point, query) index pairs for every point within r of each query centre."""
    hits = tree.query_radius(np.column_stack([qx, qy]), r=r)
    return np.concatenate(hits), np.repeat(np.arange(len(hits)), [len(h) for h in hits])


def count_balltree(tree, points, x, y, bearing, shape="circle"):
    counts = np.zeros(len(points), np.int64)
    for s in _chunks(len(x)):
        cx, cy, b = x[s], y[s], bearing[s]
        j, o = _pairs(tree, *query_disc(shape, cx, cy, b))
        if shape != "circle":                                   # the circle needs no test beyond the query
            j = j[TESTS[shape](points[j, 0], points[j, 1], cx[o], cy[o], b[o])]
        counts += np.bincount(j, minlength=len(points))
    return counts


def count_exhaustive(points, x, y, bearing, shape="circle"):
    """Same counts, testing every point against every position."""
    px, py = points[:, 0], points[:, 1]
    counts = np.zeros(len(points), np.int64)
    for s in _chunks(len(x), 128):
        counts += TESTS[shape](px, py, x[s, None], y[s, None], bearing[s, None]).sum(axis=0)
    return counts


def traffic_flow(tree, x, y, trip_codes, radius=50.0):
    """Distinct trips passing within `radius` of each point: each vehicle counts once, whatever its heading."""
    pairs = []
    for s in _chunks(len(x)):
        j, o = _pairs(tree, x[s], y[s], radius)
        pairs.append(np.column_stack([j, trip_codes[s][o]]))
    pairs = np.unique(np.vstack(pairs), axis=0)                 # one (point, trip) pair per passing trip
    return np.bincount(pairs[:, 0], minlength=tree.data.shape[0])
