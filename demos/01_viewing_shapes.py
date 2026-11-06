"""Demo 01: which circle best approximates a 60°, 100 m forward view?

Checks: of seven circles tangent at the vehicle, the one reaching 100 m ahead has the highest Dice similarity
(2|A∩B| / (|A|+|B|), Monte Carlo) with the sector.
"""
import _path  # noqa: F401
import numpy as np
from matplotlib.patches import Circle, Polygon, Wedge

from visibility_demo import shapes as S
from visibility_demo.plotting import INK, MUTED, SHAPE_COLOURS, SHAPE_STYLES, dot, legend_below, line, plt, save

REACH = [80, 90, 95, 100, 105, 110, 115]             # metres ahead reached by each candidate circle
rng = np.random.default_rng(0)
px, py = rng.uniform(-75, 75, 1_000_000), rng.uniform(0, 150, 1_000_000)   # vehicle at (0, 0), heading north
box = 150.0 ** 2
sector = S.in_sector(px, py, 0.0, 0.0, 0.0)


def dice(reach):
    r = reach / 2
    circle = px ** 2 + (py - r) ** 2 <= r ** 2
    return 2 * (sector & circle).mean() * box / (sector.mean() * box + np.pi * r ** 2)


d = {k: dice(k) for k in REACH}
best = max(d, key=d.get)
assert best == 100, d
print("Dice:", {k: round(v, 3) for k, v in d.items()})

fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.1]})
style = lambda s, lw=2.2: dict(facecolor="none", edgecolor=SHAPE_COLOURS[s], linestyle=SHAPE_STYLES[s], linewidth=lw)
a0.add_patch(Wedge((0, 0), S.RANGE_M, 90 - S.OPENING_DEG / 2, 90 + S.OPENING_DEG / 2, **style("sector")))
a0.add_patch(Polygon(np.array(S.triangle_vertices(0.0, 0.0, 0.0)), **style("triangle")))
a0.add_patch(Circle((0, S.CIRCLE_OFFSET), S.CIRCLE_R, **style("circle", 2.6)))
a0.plot(0, 0, marker="^", markersize=11, color=INK)
a0.set(xlim=(-70, 70), ylim=(-8, 112), aspect="equal", xlabel="Lateral distance (m)", ylabel="Distance ahead (m)",
       title="(a) Viewing shapes: 100 m reach, 60° opening")
a0.grid(linewidth=0.5, color="0.9")

v = np.array(list(d.values()))
a1.plot(REACH, v, color=SHAPE_COLOURS["circle"], marker="o", linewidth=2)
a1.plot(best, d[best], marker="o", markersize=13, markerfacecolor="none", markeredgecolor=INK)
a1.annotate(f"best: reaches {best} m ahead\nDice {d[best]:.3f}", (best, d[best]), xytext=(-175, -70),
            textcoords="offset points", fontsize=10, arrowprops=dict(arrowstyle="-", color=MUTED))
a1.set(ylim=(v.min() - 0.02, v.max() + 0.02), xlabel="Circle reach ahead of the vehicle (m)",
       ylabel="Dice similarity with the sector", title="(b) Circles tangent at the vehicle vs the 60° sector")
a1.grid(axis="y", linewidth=0.5, color="0.9")

legend_below(fig, [line(s, SHAPE_COLOURS[s], linestyle=SHAPE_STYLES[s], linewidth=2.2) for s in S.SHAPES]
             + [dot("vehicle (heading up)", INK, 9, marker="^")])
save(fig, "01_viewing_shapes.png")
