"""Demo 03: exposure counts from one trip, seen from the driver's position.

Every 5 s the viewing circle is placed 50 m ahead of the vehicle along its bearing, so it turns with the vehicle and
covers what lies in front of the driver. Façade points inside get one count per position; while the vehicle waits, the
same façades keep counting (time in view, not passing vehicles).
Checks: BallTree counts equal the exhaustive test.
"""
import _path  # noqa: F401
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Circle, Rectangle

from visibility_demo import exposure as E
from visibility_demo import shapes as S
from visibility_demo.data import load_scene
from visibility_demo.geo import ahead
from visibility_demo.plotting import INK, draw_buildings, draw_roads, dot, legend_below, line, map_axes, plt, ring, save

sc = load_scene()
t = sc.trips
per_trip = t.groupby("trip_id").agg(n=("x", "size"), stopped=("speed_mps", lambda v: (v == 0).sum()))
trip_id = per_trip[per_trip.n.between(60, 90)].stopped.idxmax()          # typical length, most waiting
trip = t[t.trip_id == trip_id]
x, y, b = (trip[c].to_numpy(float) for c in ("x", "y", "bearing"))
v = trip.speed_mps.to_numpy()
counts = E.count_balltree(sc.tree, sc.points, x, y, b)
assert np.array_equal(counts, E.count_exhaustive(sc.points, x, y, b))
seen = counts > 0

# three moving positions 10 s apart with the largest change of heading: the circles visibly turn with the vehicle
GAP = 2                                                                    # positions between steps (10 s)
turn = lambda i: abs((b[i + 2 * GAP] - b[i] + 180) % 360 - 180) if (v[i:i + 2 * GAP + 1] > 2).all() else -1
k0 = max(range(len(x) - 2 * GAP), key=turn)
steps = [k0, k0 + GAP, k0 + 2 * GAP]
print(f"trip {trip_id}: {len(x)} positions, {seen.sum()} points reached, {counts.sum()} exposures; "
      f"zoom: heading turns {turn(k0):.0f}° over 10 s")

NAVY, RED = "#004488", "#BB5566"
STEP_COLOURS = ["#6baed6", "#2b7bba", "#08306b"]                          # one blue per step, light -> dark
fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 6.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]})

# (a) whole trip: heading at every position, exposure from this trip
draw_roads(a0, sc.roads)
draw_buildings(a0, sc.rings)
a0.scatter(*sc.points[~seen].T, s=1, color="#b9c0c8", linewidths=0, zorder=2)
o = np.argsort(counts[seen])
norm = BoundaryNorm([1, 2, 3, 5, 8, 13, 21, max(22, counts.max() + 1)], plt.get_cmap("plasma").N)
pts = a0.scatter(*sc.points[seen][o].T, c=counts[seen][o], s=6, cmap="plasma", norm=norm, linewidths=0, zorder=3)
a0.plot(x, y, color=INK, linewidth=1.2, zorder=4)
hx, hy = ahead(0.0, 0.0, b, 1.0)
a0.quiver(x, y, hx, hy, color=INK, scale=40, width=0.004, headwidth=4, zorder=5)    # bearing at each 5-s position
stop = v == 0
a0.scatter(x[stop], y[stop], s=55, facecolors="none", edgecolors=RED, linewidths=1.5, zorder=6)
pad = 120
x0, x1, y0, y1 = x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad
cxy = np.array([S.circle_centre(x[k], y[k], b[k]) for k in steps] + [(x[k], y[k]) for k in steps])
lo, hi = cxy.min(axis=0) - S.CIRCLE_R - 15, cxy.max(axis=0) + S.CIRCLE_R + 15
(zx, zy), Z = (lo + hi) / 2, (hi - lo).max() / 2                           # square zoom window around all circles
a0.add_patch(Rectangle((zx - Z, zy - Z), 2 * Z, 2 * Z, fill=False, edgecolor=NAVY, linewidth=1.8, zorder=7))
map_axes(a0, x0, x1, y0, y1, bar=200)
a0.set_title(f"(a) One trip: {len(x)} positions, {counts.sum():,} exposures\n      to {seen.sum():,} façade points")
fig.colorbar(pts, ax=a0, fraction=0.04, pad=0.01, label="Exposures from this trip (count; 1 count = 5 s in view)")

# (b) driver's view at three positions 10 s apart
draw_roads(a1, sc.roads)
draw_buildings(a1, sc.rings)
inside = np.array([S.in_circle(sc.points[:, 0], sc.points[:, 1], x[k], y[k], b[k]) for k in steps])
hits = inside.sum(axis=0)
a1.scatter(*sc.points[hits == 0].T, s=4, color="#b9c0c8", linewidths=0, zorder=2)
a1.plot(x, y, color=INK, linewidth=1.0, alpha=0.5, zorder=3)
for n, (k, col) in enumerate(zip(steps, STEP_COLOURS)):
    cx, cy = S.circle_centre(x[k], y[k], b[k])
    a1.add_patch(Circle((cx, cy), S.CIRCLE_R, fill=False, edgecolor=col, linewidth=2.4, zorder=4))
    a1.plot([x[k], cx], [y[k], cy], color=col, linewidth=1.2, linestyle=(0, (3, 2)), zorder=4)   # 50 m ahead
    a1.plot(cx, cy, marker="+", color=col, markersize=10, markeredgewidth=2, zorder=5)
    fx, fy = ahead(x[k], y[k], b[k], 22.0)
    a1.annotate("", xy=(fx, fy), xytext=(x[k], y[k]), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=2, mutation_scale=16))
    a1.plot(x[k], y[k], marker="o", color=col, markeredgecolor=INK, markersize=10, zorder=7)
    lx, ly = ahead(x[k], y[k], b[k] + 90, 14)                              # label beside the vehicle, to its right
    a1.text(lx, ly, f"t = {10 * n} s", color=col, fontsize=11, fontweight="bold", ha="center", va="center", zorder=8,
            bbox=dict(facecolor="white", edgecolor=col, pad=1.5, alpha=0.9))
hit_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], plt.get_cmap("plasma").N)
sc_hits = a1.scatter(*sc.points[hits > 0].T, c=hits[hits > 0], s=16, cmap="plasma", norm=hit_norm, linewidths=0,
                     zorder=3)
map_axes(a1, zx - Z, zx + Z, zy - Z, zy + Z, bar=50)
a1.set_title(f"(b) Driver's view at 10 s intervals: heading turns {turn(k0):.0f}°,\n      the viewing circle turns with it")
cb = fig.colorbar(sc_hits, ax=a1, fraction=0.04, pad=0.01, ticks=[1, 2, 3])
cb.set_label("Circles containing the point (of these three)")

legend_below(fig, [line("trip; arrow = bearing (direction of travel)", INK, marker=">", markersize=6, linewidth=1.2),
                   ring("stopped position", RED, 8),
                   line("viewing circle: 50 m radius, centre 50 m ahead (+)", NAVY, linewidth=2.2),
                   dot("façade point not in view", "#b9c0c8", 4)], ncol=2)
save(fig, "03_one_trip.png")
