"""Demo 02: the inputs - buildings, roads, façade points and vehicle positions.

Checks: the share of positions within 20 m of a major road exceeds the major roads' share of network length.
"""
import _path  # noqa: F401
import numpy as np
from matplotlib.colors import LogNorm
from sklearn.neighbors import BallTree

from visibility_demo.buildings import densify
from visibility_demo.data import load_scene
from visibility_demo.plotting import draw_buildings, dot, legend_below, line, map_axes, plt, save
from visibility_demo.synthetic import MAJOR, road_class

sc = load_scene()
x, y, _ = sc.xyb
major = [xy for cls, xy in sc.roads if road_class(cls) in MAJOR]
length = lambda xy: np.hypot(*np.diff(xy, axis=0).T).sum()
share_len = sum(map(length, major)) / sum(length(xy) for _, xy in sc.roads)
dist, _ = BallTree(np.vstack([densify(np.vstack([xy, xy[-1:]]), 2.0) for xy in major])).query(np.column_stack([x, y]))
share_pos = (dist[:, 0] <= 20).mean()
assert share_pos > share_len, (share_pos, share_len)
print(f"major roads: {share_len:.0%} of length, {share_pos:.0%} of positions within 20 m")

BLUE, GREY = "#2F629C", "#9aa4ae"
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 5.6), layout="constrained")
draw_buildings(a0, sc.rings)
for cls, xy in sc.roads:
    is_major = road_class(cls) in MAJOR
    a0.plot(*xy.T, color=BLUE if is_major else GREY, linewidth=1.8 if is_major else 0.8, zorder=2)
a0.scatter(*sc.points.T, s=0.3, color="#46515e", linewidths=0, zorder=3)
map_axes(a0, *sc.bounds(), bar=500)
a0.set_title(f"(a) {len(sc.rings):,} buildings, {len(sc.points):,} façade points")

hb = a1.hexbin(x, y, gridsize=90, extent=sc.bounds(), norm=LogNorm(), cmap="viridis", mincnt=1, zorder=2)
draw_buildings(a1, sc.rings, fc="#f4f6f8", ec="#e2e6ea")
map_axes(a1, *sc.bounds(), bar=500)
a1.set_title(f"(b) {sc.trips.trip_id.nunique():,} synthetic trips, {len(x):,} positions")
fig.colorbar(hb, ax=a1, fraction=0.04, pad=0.01, label="Five-second positions per hexagon (log scale)")

legend_below(fig, [line("major road (motorway to tertiary)", BLUE, linewidth=2), line("minor road", GREY, linewidth=1),
                   dot("façade point (≤ 10 m apart)", "#46515e", 3)])
save(fig, "02_data_overview.png")
