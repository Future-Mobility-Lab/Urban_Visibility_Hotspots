"""Demo 07: BallTree vs exhaustive testing for the three viewing shapes.

Both methods are vectorised alike; only candidate retrieval differs. Medians of three runs on this machine.
Checks: identical counts and a speed-up for every shape; the circle is the fastest shape with the BallTree.

    python demos/07_benchmark.py [--trips 150]
"""
import argparse
import time

import _path  # noqa: F401
import numpy as np
import pandas as pd

from visibility_demo import exposure as E
from visibility_demo.data import load_scene, output
from visibility_demo.plotting import CHROME, INK, SHAPE_COLOURS, legend_below, line, plt, save
from visibility_demo.shapes import SHAPES

ap = argparse.ArgumentParser()
ap.add_argument("--trips", type=int, default=150)
n_trips = ap.parse_args().trips

sc = load_scene()
sub = sc.trips[sc.trips.trip_code < n_trips]
x, y, b = (sub[c].to_numpy(float) for c in ("x", "y", "bearing"))
pts = sc.points


def timed(fn, reps=3):
    """(median seconds, last result)"""
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return np.median(times), out


build_s, tree = timed(lambda: E.build_index(pts))
rows = []
for s in SHAPES:
    t_tree, c_tree = timed(lambda: E.count_balltree(tree, pts, x, y, b, s))
    t_all, c_all = timed(lambda: E.count_exhaustive(pts, x, y, b, s))
    assert np.array_equal(c_tree, c_all), s
    rows.append(dict(shape=s, balltree_ms=1000 * t_tree / n_trips, exhaustive_ms=1000 * t_all / n_trips,
                     speedup=t_all / t_tree))
    print(f"{s:9s} {rows[-1]['balltree_ms']:.2f} vs {rows[-1]['exhaustive_ms']:.1f} ms per trip ({rows[-1]['speedup']:.0f}x)")
res = pd.DataFrame(rows)
res.to_csv(output("benchmark.csv"), index=False)
assert (res.speedup > 1).all() and res["shape"][res.balltree_ms.idxmin()] == "circle"

fig, ax = plt.subplots(figsize=(8.5, 4.4), layout="constrained")
hgt = 0.36
for k, r in res.iterrows():
    col = SHAPE_COLOURS[r["shape"]]
    ax.barh(k - hgt / 2, r.balltree_ms, height=hgt, color=col, zorder=3)
    ax.barh(k + hgt / 2, r.exhaustive_ms, height=hgt, color=col, alpha=0.35, hatch="//", edgecolor=col, linewidth=0, zorder=3)
    ax.text(r.balltree_ms * 1.12, k - hgt / 2, f"{r.balltree_ms:.2f} ms", va="center", fontsize=10)
    ax.text(r.exhaustive_ms * 1.12, k + hgt / 2, f"{r.exhaustive_ms:.1f} ms", va="center", fontsize=10)
    ax.text(1.0, k, f"{r.speedup:.0f}×", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=13,
            fontweight="bold", color=INK)
ax.set_yticks(range(len(res)), [s.capitalize() for s in res["shape"]])
ax.invert_yaxis()
ax.set(xscale="log", xlim=(res.balltree_ms.min() / 2, res.exhaustive_ms.max() * 12),
       xlabel=f"Milliseconds per trip (log scale), first {n_trips} trips, median of 3 runs",
       title=f"BallTree vs testing every point: identical counts; index built once in {build_s * 1000:.0f} ms")
ax.grid(axis="x", linewidth=0.5, color="0.9")
ax.set_axisbelow(True)
legend_below(fig, [line("BallTree", CHROME, linewidth=8), line("every point tested (hatched)", CHROME, linewidth=8, alpha=0.4)])
save(fig, "07_benchmark.png")
