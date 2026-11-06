"""Demo 04: how a BallTree answers one viewing-circle query.

Built once: points are split recursively and each group wrapped in its smallest enclosing ball. A query walks the tree:
balls missing the circle are skipped whole, balls inside it are accepted whole, only leaves crossing its edge are tested.
Checks: the walk returns exactly tree.query_radius and the exhaustive result; under 5 % of points are tested.
"""
import _path  # noqa: F401
import numpy as np
from matplotlib.patches import Circle, Rectangle

from visibility_demo import shapes as S
from visibility_demo.data import load_scene
from visibility_demo.plotting import CHROME, INK, dot, legend_below, line, map_axes, plt, ring, save

sc = load_scene()
pts, tree = sc.points, sc.tree
_, order, nodes, bounds = tree.get_arrays()
centre, radius, is_leaf = bounds[0], nodes["radius"], nodes["is_leaf"].astype(bool)


def members(i):
    return order[nodes["idx_start"][i]:nodes["idx_end"][i]]


def walk(q, r, i=0, out=None):
    """Classify tree nodes for a radius query: lists of skipped, accepted (wholly inside) and opened leaves."""
    out = out or {"skipped": [], "accepted": [], "opened": []}
    d = np.hypot(*(centre[i] - q))
    if d - radius[i] > r:
        out["skipped"].append(i)
    elif d + radius[i] <= r:
        out["accepted"].append(i)
    elif is_leaf[i]:
        out["opened"].append(i)
    else:
        walk(q, r, 2 * i + 1, out)
        walk(q, r, 2 * i + 2, out)
    return out


x, y, b = sc.xyb
mid = np.median(x), np.median(y)
moving = np.flatnonzero(sc.trips.speed_mps.to_numpy() > 5)
k = moving[np.argmin(np.hypot(x[moving] - mid[0], y[moving] - mid[1]))]   # a moving vehicle near the centre
q, R = np.array(S.circle_centre(x[k], y[k], b[k])), S.CIRCLE_R
w = walk(q, R)
tested = np.concatenate([members(i) for i in w["opened"]])
found = np.union1d(tested[np.hypot(*(pts[tested] - q).T) <= R],
                   np.concatenate([members(i) for i in w["accepted"]] or [np.array([], int)]))
assert set(found) == set(tree.query_radius(q[None], r=R)[0])
assert set(found) == set(np.flatnonzero(S.in_circle(pts[:, 0], pts[:, 1], x[k], y[k], b[k])))
assert len(tested) < 0.05 * len(pts)
n_leaves = is_leaf.sum()
print(f"{len(w['opened'])} of {n_leaves} leaves opened, {len(tested)} of {len(pts):,} points tested, {len(found)} inside")

BLUE, PINK, NAVY = "#2F629C", "#A82A78", "#004488"
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.05]})
a0.scatter(*pts.T, s=0.3, color=CHROME, linewidths=0)
for depth, lw, al in ((1, 2.4, 0.9), (3, 1.4, 0.7), (5, 0.8, 0.5)):
    for i in range(2 ** depth - 1, min(2 ** (depth + 1) - 1, len(radius))):
        a0.add_patch(Circle(centre[i], radius[i], fill=False, edgecolor=BLUE, linewidth=lw, alpha=al))
h = 260                                                                   # half-width of the zoom window (m)
a0.add_patch(Rectangle(q - h, 2 * h, 2 * h, fill=False, edgecolor=PINK, linewidth=2, zorder=4))
map_axes(a0, pts[:, 0].min() - 50, pts[:, 0].max() + 50, pts[:, 1].min() - 50, pts[:, 1].max() + 50, bar=500)
a0.set_title("(a) Built once: façade points in nested balls\n      (levels 1, 3 and 5 of the tree shown)")

near = lambda p: (np.abs(p - q) < h + 80).all(axis=1)
a1.scatter(*pts[near(pts)].T, s=5, color=CHROME, linewidths=0)
for i in np.flatnonzero(is_leaf & near(centre)):
    opened = i in w["opened"]
    a1.add_patch(Circle(centre[i], radius[i], fill=False, edgecolor=BLUE if opened else "#c9d0d8",
                        linewidth=1.8 if opened else 0.8, zorder=3 if opened else 2))
a1.scatter(*pts[tested].T, s=9, color=BLUE, linewidths=0, zorder=4)
a1.scatter(*pts[found].T, s=12, color=PINK, linewidths=0, zorder=5)
a1.add_patch(Circle(q, R, fill=False, edgecolor=NAVY, linewidth=2.4, zorder=6))
a1.annotate("", xy=q, xytext=(x[k], y[k]), arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.6), zorder=7)
a1.plot(x[k], y[k], marker="o", color=INK, markersize=8, zorder=7)
map_axes(a1, q[0] - h, q[0] + h, q[1] - h, q[1] + h, bar=100)
a1.set_title(f"(b) One query: {len(w['opened'])} of {n_leaves} leaves searched,\n"
             f"      {len(tested)} of {len(pts):,} points tested, {len(found)} inside")

legend_below(fig, [dot("façade point", CHROME, 4), ring("leaf skipped", "#c9d0d8", 10, 1.2),
                   ring("leaf searched", BLUE, 10, 1.8), dot("point tested", BLUE, 4),
                   dot("point inside: counted", PINK, 4), line("viewing circle", NAVY, linewidth=2.4),
                   dot("vehicle (arrow: heading)", INK, 6)], ncol=4)
save(fig, "04_balltree_query.png")
