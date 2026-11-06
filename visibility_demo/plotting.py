"""Shared figure style and map helpers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection      # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402

from .data import FIGURES                                               # noqa: E402

SHAPE_COLOURS = {"circle": "#004488", "sector": "#DDAA33", "triangle": "#BB5566"}   # Paul Tol, CVD-safe
SHAPE_STYLES = {"circle": "-", "sector": (0, (6, 2)), "triangle": (0, (2, 1.5))}
INK, MUTED, CHROME = "#1b2733", "#46515e", "#9aa4ae"
ROAD_WIDTH = {"motorway": 3.0, "trunk": 3.0, "primary": 2.4, "secondary": 2.0, "tertiary": 1.6}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def draw_buildings(ax, rings, fc="#e9ecef", ec="#c3cad2", zorder=1):
    ax.add_collection(PolyCollection(rings, facecolor=fc, edgecolor=ec, linewidth=0.4, zorder=zorder))


def draw_roads(ax, roads, colour="#d5dae0", zorder=0.5):
    for cls, xy in roads:
        ax.add_collection(LineCollection([xy], colors=colour, linewidths=ROAD_WIDTH.get(cls.removesuffix("_link"), 1.0),
                                         capstyle="round", zorder=zorder))


def map_axes(ax, x0, x1, y0, y1, bar=None):
    """Equal-aspect map frame without ticks, with an optional scale bar of `bar` metres."""
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal", xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor("#c9d0d8")
    if bar:
        bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.05 * (y1 - y0)
        ax.plot([bx, bx + bar], [by, by], color=INK, linewidth=3, solid_capstyle="butt", zorder=9)
        ax.text(bx + bar / 2, by + 0.02 * (y1 - y0), f"{bar} m", ha="center", va="bottom", fontsize=10, color=INK,
                zorder=9, bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.8))


def line(label, color, **kw):
    return Line2D([], [], color=color, label=label, **kw)


def dot(label, color, size=5, **kw):
    return Line2D([], [], marker=kw.pop("marker", "o"), linestyle="", color=color, markersize=size, label=label, **kw)


def ring(label, color, size=9, width=1.6):
    return Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=color, markersize=size,
                  markeredgewidth=width, label=label)


def legend_below(fig, handles, ncol=None):
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol or len(handles), frameon=False)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote figures/{name}")
