"""Demo 05: exposure of every façade point over all trips, and its concentration.

Writes outputs/exposure_circle.csv. The concentration curve sorts points from most to least exposed against their
cumulative share of exposure; above the diagonal = a minority of points takes most of the exposure.
Checks: the top 10 % of points seen receive more than 10 % of exposure.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from visibility_demo import SAMPLE_INTERVAL_S
from visibility_demo import exposure as E
from visibility_demo.data import PROJ, load_scene, output
from visibility_demo.plotting import CHROME, INK, dot, draw_buildings, draw_roads, legend_below, line, map_axes, plt, save

sc = load_scene()
counts = E.count_balltree(sc.tree, sc.points, *sc.xyb)
minutes = counts * SAMPLE_INTERVAL_S / 60
lon, lat = PROJ.to_lonlat(*sc.points.T)
pd.DataFrame({"point": np.arange(len(counts)), "building": sc.owner, "lon": lon.round(7), "lat": lat.round(7),
              "exposures": counts, "minutes_in_view": minutes.round(2)}).to_csv(output("exposure_circle.csv"), index=False)

seen = np.sort(counts[counts > 0])[::-1]
cum = np.cumsum(seen) / seen.sum()
frac = np.arange(1, len(seen) + 1) / len(seen)
share = {q: cum[round(q * len(seen)) - 1] for q in (0.01, 0.10, 0.50)}
assert share[0.10] > 0.10
print(f"{len(seen):,} of {len(counts):,} points seen; top 1/10/50 % -> "
      + " / ".join(f"{v:.0%}" for v in share.values()) + f" of exposure; max {minutes.max():.0f} min")

PINK = "#A82A78"
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 5.6), layout="constrained", gridspec_kw={"width_ratios": [1.35, 1]})
draw_roads(a0, sc.roads, colour="#e3e7eb")
draw_buildings(a0, sc.rings, fc="#f4f6f8", ec="#dde2e7")
a0.scatter(*sc.points[counts == 0].T, s=0.5, color="#c9d0d8", linewidths=0, zorder=2)
on = np.argsort(minutes)
on = on[counts[on] > 0]                                                  # hottest drawn last
im = a0.scatter(*sc.points[on].T, c=minutes[on], s=2.5, cmap="plasma", norm=LogNorm(minutes[on].min(), minutes.max()),
                linewidths=0, zorder=3)
map_axes(a0, *sc.bounds(), bar=500)
a0.set_title("(a) Time in view per façade point, all synthetic trips")
fig.colorbar(im, ax=a0, fraction=0.04, pad=0.01, label="Minutes in view (log scale)")

a1.plot(frac * 100, cum * 100, color=PINK, linewidth=2.4, zorder=3)
a1.plot([0, 100], [0, 100], color=CHROME, linewidth=1, linestyle=(0, (4, 3)))
for (q, v), mk in zip(share.items(), "Dos"):
    a1.plot(q * 100, v * 100, marker=mk, color=INK, markersize=7, zorder=4)
    a1.annotate(f"top {q:.0%}: {v:.0%}", (q * 100, v * 100), xytext=(12, -14), textcoords="offset points", fontsize=10)
a1.set(xlim=(0, 100), ylim=(0, 102), xlabel="Façade points, most exposed first (% of points seen)",
       ylabel="Cumulative share of all exposure (%)", title="(b) Concentration of exposure")
a1.grid(linewidth=0.5, color="0.9")

legend_below(fig, [line("concentration curve", PINK, linewidth=2.4),
                   line("equal exposure (diagonal)", CHROME, linewidth=1, linestyle=(0, (4, 3))),
                   dot("façade point never in view", "#c9d0d8", 4)])
save(fig, "05_exposure_map.png")
