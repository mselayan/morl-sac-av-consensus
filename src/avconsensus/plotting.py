"""Figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def plot_gmm_threshold(M, gmm, threshold, path):
    """Histogram of positive GSSM risk with GMM components and threshold."""
    pos = M[np.isfinite(M) & (M > 0)]
    x = np.linspace(pos.min(), pos.max(), 500)
    order = np.argsort(gmm.means_.ravel())
    labels = ["Routine", "Moderate", "Elevated"] if len(order) == 3 else [f"C{k}" for k in order]
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos, bins=100, density=True, color="#00a5e3", alpha=0.6, label="M distribution")
    for k, lab, col in zip(order, labels, colors):
        w, m, sd = gmm.weights_[k], gmm.means_[k, 0], np.sqrt(gmm.covariances_[k].ravel()[0])
        ax.plot(x, w * norm.pdf(x, m, sd), color=col, lw=2, label=lab)
    ax.axvline(threshold, color="#740e4c", ls="--", lw=2, label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("M (GSSM risk)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_surface(surface, composites, path, axis_min=0.4, cloud_sample=3000,
                        view=None):
    """3D Pareto frontier with upper and lower hull facets and the
    non-Pareto cloud colored by dataset."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    view = view or {"elev": 22, "azim": 140}
    pts, hull = surface["pareto_points"], surface["c_hull_model"]
    upper = hull.equations[:, :3].sum(axis=1) > 0
    obj = ["Safety", "Efficiency", "Interaction"]

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    cloud = composites[~composites["pareto_flag"]]
    cloud = cloud[(cloud[obj] >= axis_min).all(axis=1)]
    for name, color, alpha in [("FB", "#8968CD", 0.3), ("I-395", "#FFBF65", 0.6)]:
        c = cloud[cloud["dataset"] == name]
        if len(c):
            c = c.sample(n=min(cloud_sample, len(c)), random_state=42)
            ax.scatter(*c[obj].to_numpy().T, c=color, s=2, alpha=alpha,
                       label="Foggy Bottom" if name == "FB" else name)

    if (~upper).any():
        ax.add_collection3d(Poly3DCollection(pts[hull.simplices[~upper]], alpha=0.15,
                                             facecolor="#6C88C4", edgecolor="#6C88C4",
                                             linewidth=0.5))
    ax.add_collection3d(Poly3DCollection(pts[hull.simplices[upper]], alpha=0.6,
                                         facecolor="#8DD7BF", edgecolor="#8DD7BF",
                                         linewidth=0.8))
    ax.scatter(*pts.T, c="#E77577", s=40, edgecolors="black", linewidth=0.2,
               label="Pareto-optimal")
    ax.scatter([1], [1], [1], c="black", s=170, marker="*", edgecolors="white",
               linewidth=0.8, label="Utopia (1, 1, 1)")

    ax.set_xlabel("\nSafety ($S$)", fontsize=18, labelpad=12)
    ax.set_ylabel("\nEfficiency ($E$)", fontsize=18, labelpad=12)
    ax.set_zlabel("")
    ax.text2D(0.02, 0.5, "Interaction ($I$)", fontsize=18, transform=ax.transAxes,
              rotation=90, ha="center", va="center")
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(axis_min, 1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#DDDDDD")
    ax.tick_params(labelsize=17, pad=6)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=16, framealpha=0.05, edgecolor="#CCCCCC", loc="upper center",
              bbox_to_anchor=(0.5, 0.94), ncol=4, columnspacing=1.0)
    ax.view_init(**view)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
