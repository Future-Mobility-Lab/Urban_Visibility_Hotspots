"""Demo 08: do circle, sector and triangle rank façades alike?

Compares exposure per point across shapes: Pearson r (as in the paper) and overlap of the top 10 % of points.
Checks: every pairwise Pearson r exceeds 0.8.
"""
import _path  # noqa: F401
import numpy as np
from matplotlib.colors import LogNorm

from visibility_demo import exposure as E
from visibility_demo.data import load_scene
from visibility_demo.plotting import INK, SHAPE_COLOURS, dot, legend_below, line, plt, save
from visibility_demo.shapes import SHAPES

sc = load_scene()
C = {s: E.count_balltree(sc.tree, sc.points, *sc.xyb, shape=s) for s in SHAPES}
R = np.corrcoef([C[s] for s in SHAPES])
top = {s: set(np.argsort(-C[s])[:int(0.1 * (C[s] > 0).sum())]) for s in SHAPES}
overlap = lambda a, b: len(top[a] & top[b]) / len(top[a])
assert R[np.triu_indices(3, 1)].min() > 0.8, R
print("Pearson r:\n", R.round(3))

labels = [s.capitalize() for s in SHAPES]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained", gridspec_kw={"width_ratios": [0.9, 1, 1]})
im = axes[0].imshow(R, cmap="Blues", vmin=0.5, vmax=1)
for (i, j), r in np.ndenumerate(R):
    axes[0].text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=12, fontweight="bold",
                 color="white" if r > 0.85 else INK)
axes[0].set(xticks=range(3), xticklabels=labels, yticks=range(3), yticklabels=labels,
            title="(a) Pearson r of counts per point")
axes[0].spines[:].set_visible(False)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03, label="Pearson r")

for ax, other, tag in ((axes[1], "sector", "b"), (axes[2], "triangle", "c")):
    m = (C["circle"] > 0) | (C[other] > 0)
    ax.hexbin(C["circle"][m] + 1, C[other][m] + 1, gridsize=40, xscale="log", yscale="log", norm=LogNorm(), cmap="Greys",
              mincnt=1)
    lim = max(C["circle"].max(), C[other].max()) + 1
    ax.plot([1, lim], [1, lim], color=SHAPE_COLOURS[other], linewidth=1, linestyle=(0, (4, 3)), alpha=0.8)
    ax.set(xlabel="Circle: exposures + 1 (count, log scale)", ylabel=f"{other.capitalize()}: exposures + 1 (count, log scale)",
           title=f"({tag}) Circle vs {other}: top-10 % overlap {overlap('circle', other):.0%}")

legend_below(fig, [line(f"equal counts (circle = {s})", SHAPE_COLOURS[s], linewidth=1, linestyle=(0, (4, 3)))
                   for s in ("sector", "triangle")] + [dot("façade points per hexagon (darker = more)", "#6b7580", 8, marker="h")])
save(fig, "08_shape_agreement.png")
