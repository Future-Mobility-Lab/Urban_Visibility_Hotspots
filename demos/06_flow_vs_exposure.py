"""Demo 06: exposure is not traffic flow.

Flow counts each trip passing within 50 m once; exposure counts 5-s intervals in a vehicle's forward view, so it
depends on heading and on how long vehicles stay.
Checks: Spearman rho between 0 and 1; among points with equal flow, exposure differs more than twofold.
"""
import _path  # noqa: F401
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

from visibility_demo import SAMPLE_INTERVAL_S
from visibility_demo import exposure as E
from visibility_demo.data import load_scene
from visibility_demo.plotting import INK, draw_buildings, draw_roads, legend_below, map_axes, plt, ring, save

sc = load_scene()
x, y, b = sc.xyb
expo = E.count_balltree(sc.tree, sc.points, x, y, b)
flow = E.traffic_flow(sc.tree, x, y, sc.trips.trip_code.to_numpy())
minutes = expo * SAMPLE_INTERVAL_S / 60
m = (flow > 0) & (expo > 0)
rho = spearmanr(flow[m], expo[m]).statistic
assert 0 < rho < 1


def widest_pair(idx):
    """Among points with identical flow, the pair with the largest exposure ratio: (ratio, high, low, flow)."""
    best = (0, None, None, None)
    for f in np.unique(flow[idx]):
        same = idx[flow[idx] == f]
        hi, lo = same[np.argmax(expo[same])], same[np.argmin(expo[same])]
        best = max(best, (expo[hi] / expo[lo], hi, lo, int(f)), key=lambda t: t[0])
    return best


ratio, A, B, f_ab = widest_pair(np.flatnonzero(m & (flow >= np.percentile(flow[m], 75))))   # busy points only
assert ratio > 2
print(f"Spearman rho {rho:.2f}; A and B both passed by {f_ab} trips: {minutes[A]:.0f} vs {minutes[B]:.0f} min ({ratio:.1f}x)")

RED = "#BB5566"
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.1]})
hb = a0.hexbin(flow[m], minutes[m], gridsize=45, xscale="log", yscale="log", norm=LogNorm(), cmap="viridis", mincnt=1)
for p, label in ((A, "A"), (B, "B")):
    a0.plot(flow[p], minutes[p], marker="o", markersize=11, markerfacecolor="none", markeredgecolor=RED,
            markeredgewidth=2.2)
    a0.annotate(label, (flow[p], minutes[p]), xytext=(8, 4), textcoords="offset points", fontsize=12, fontweight="bold",
                color=RED)
a0.set(xlabel="Traffic flow: trips passing within 50 m (count, log scale)", ylabel="Exposure: minutes in view (log scale)",
       title=f"(a) Façade points: flow vs exposure (Spearman ρ = {rho:.2f})")
fig.colorbar(hb, ax=a0, fraction=0.045, pad=0.01, label="Façade points per hexagon (log scale)")

c = (sc.points[A] + sc.points[B]) / 2
h = max(380, np.abs(sc.points[A] - sc.points[B]).max() / 2 + 150)
draw_roads(a1, sc.roads)
draw_buildings(a1, sc.rings)
on = np.argsort(minutes)
on = on[expo[on] > 0]
im = a1.scatter(*sc.points[on].T, c=minutes[on], s=4, cmap="plasma", norm=LogNorm(minutes[on].min(), minutes.max()),
                linewidths=0, zorder=3)
for p, label in ((A, "A"), (B, "B")):
    a1.plot(*sc.points[p], marker="o", markersize=14, markerfacecolor="none", markeredgecolor=RED, markeredgewidth=2.4,
            zorder=5)
    a1.annotate(f"{label}: {minutes[p]:.0f} min", sc.points[p], xytext=(10, 10), textcoords="offset points", fontsize=11,
                fontweight="bold", color=INK, zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
map_axes(a1, c[0] - h, c[0] + h, c[1] - h, c[1] + h, bar=200)
a1.set_title(f"(b) Same traffic ({f_ab} trips each), {ratio:.1f}× difference in exposure")
fig.colorbar(im, ax=a1, fraction=0.045, pad=0.01, label="Minutes in view (log scale)")

legend_below(fig, [ring("points A and B: identical traffic flow", RED, 10, 2)])
save(fig, "06_flow_vs_exposure.png")
