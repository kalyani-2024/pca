"""Generate all figures and one animated GIF used in PCA_Complete_Guide.md.

Run once:
    python scripts/generate_visuals.py
Outputs land in ./assets/.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch
from PIL import Image
from sklearn.datasets import make_blobs, make_circles, load_digits
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ---------- helpers ----------

def correlated_cloud(n=400, angle_deg=30, sx=3.0, sy=0.9, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 2)) * np.array([sx, sy])
    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return base @ R.T


def draw_vec(ax, origin, direction, length, color, label=None, lw=2.5,
             label_pos=0.55, label_offset=(0.15, 0.35)):
    """Draw arrow and place label near the middle of the arrow (not beyond tip)."""
    d = direction / np.linalg.norm(direction) * length
    arr = FancyArrowPatch(origin, origin + d,
                          arrowstyle="-|>", mutation_scale=18,
                          color=color, lw=lw, zorder=5)
    ax.add_patch(arr)
    if label:
        # place along the arrow at label_pos fraction, offset perpendicular a bit
        mid = origin + d * label_pos
        # perpendicular unit vector for offset
        perp = np.array([-d[1], d[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-9)
        pos = mid + perp * label_offset[1] + (d / np.linalg.norm(d)) * label_offset[0]
        ax.text(pos[0], pos[1], label, color=color,
                fontsize=11, fontweight="bold", zorder=6,
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2))


# ---------- 1. Geometric intuition: ellipse + PC1/PC2 ----------

def fig_intuition():
    X = correlated_cloud(n=500, angle_deg=30, sx=3.2, sy=0.9, seed=1)
    p = PCA(n_components=2).fit(X)
    center = p.mean_
    pc1 = p.components_[0]
    pc2 = p.components_[1]
    s1 = np.sqrt(p.explained_variance_[0])
    s2 = np.sqrt(p.explained_variance_[1])

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(X[:, 0], X[:, 1], s=14, alpha=0.45, color="#4c78a8",
               edgecolor="white", lw=0.3)
    # 2-sigma ellipse aligned with PCs
    ang = np.degrees(np.arctan2(pc1[1], pc1[0]))
    ell = Ellipse(xy=center, width=4 * s1, height=4 * s2, angle=ang,
                  edgecolor="#555", facecolor="none", lw=1.5, ls="--")
    ax.add_patch(ell)
    draw_vec(ax, center, pc1, 1.9 * s1, "#d62728", "PC1 (max variance)")
    draw_vec(ax, center, pc2, 1.9 * s2, "#2ca02c", "PC2 (⊥ PC1)",
             label_offset=(0.0, 0.8))
    ax.set_aspect("equal"); ax.set_title("PCA as rotation to variance-aligned axes")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    lim = 5.5; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.tight_layout(); fig.savefig(OUT / "01_intuition_ellipse.png"); plt.close(fig)


# ---------- 2. Animated GIF: variance as axis rotates ----------

def gif_variance_sweep():
    X = correlated_cloud(n=350, angle_deg=30, sx=3.2, sy=0.9, seed=2)
    X = X - X.mean(axis=0)
    angles = np.linspace(0, np.pi, 90)
    variances = np.array(
        [np.var(X @ np.array([np.cos(t), np.sin(t)])) for t in angles]
    )
    best_t = angles[np.argmax(variances)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.suptitle("Finding PC1: the direction that maximises projected variance",
                 fontsize=12)

    axL.scatter(X[:, 0], X[:, 1], s=12, alpha=0.4, color="#4c78a8",
                edgecolor="white", lw=0.2)
    axL.set_aspect("equal"); axL.set_xlim(-6, 6); axL.set_ylim(-6, 6)
    axL.set_xlabel("x₁"); axL.set_ylabel("x₂")
    (line_axis,) = axL.plot([], [], color="#d62728", lw=2.5)
    proj_scatter = axL.scatter([], [], s=10, color="#d62728", alpha=0.7)
    txt_var = axL.text(-5.7, 5.2, "", fontsize=11,
                       bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    axR.plot(np.degrees(angles), variances, color="#888", lw=1.2)
    axR.axvline(np.degrees(best_t), color="#d62728", ls="--", lw=1.2,
                label=f"argmax = {np.degrees(best_t):.1f}°")
    axR.set_xlabel("axis angle (deg)"); axR.set_ylabel("projected variance")
    axR.set_title("variance vs direction"); axR.legend(loc="lower center")
    (marker,) = axR.plot([], [], "o", color="#d62728", ms=8)

    def init():
        line_axis.set_data([], [])
        proj_scatter.set_offsets(np.empty((0, 2)))
        marker.set_data([], [])
        txt_var.set_text("")
        return line_axis, proj_scatter, marker, txt_var

    def update(i):
        t = angles[i]
        u = np.array([np.cos(t), np.sin(t)])
        L = 6.5
        line_axis.set_data([-L * u[0], L * u[0]], [-L * u[1], L * u[1]])
        proj_len = X @ u
        proj_pts = np.outer(proj_len, u)
        proj_scatter.set_offsets(proj_pts)
        marker.set_data([np.degrees(t)], [variances[i]])
        txt_var.set_text(f"angle = {np.degrees(t):5.1f}°\nvariance = {variances[i]:.2f}")
        return line_axis, proj_scatter, marker, txt_var

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=len(angles), interval=70, blit=True)
    gif_path = OUT / "02_variance_sweep.gif"
    anim.save(gif_path, writer=animation.PillowWriter(fps=14))
    plt.close(fig)


# ---------- 3. Projection and reconstruction error ----------

def fig_projection_reconstruction():
    X = correlated_cloud(n=120, angle_deg=25, sx=3.0, sy=1.0, seed=7)
    X = X - X.mean(axis=0)
    p = PCA(n_components=1).fit(X)
    v = p.components_[0]
    Z = X @ v
    X_hat = np.outer(Z, v)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    L = 6
    ax.plot([-L * v[0], L * v[0]], [-L * v[1], L * v[1]],
            color="#d62728", lw=2, label="PC1")
    for xi, xh in zip(X, X_hat):
        ax.plot([xi[0], xh[0]], [xi[1], xh[1]],
                color="#888", lw=0.8, alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], s=24, color="#4c78a8",
               edgecolor="white", lw=0.4, label="original")
    ax.scatter(X_hat[:, 0], X_hat[:, 1], s=18, color="#d62728",
               alpha=0.9, label="projection onto PC1")
    ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_title("PCA minimises sum of squared perpendicular distances")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "03_projection_reconstruction.png")
    plt.close(fig)


# ---------- 4. Scree + cumulative EVR ----------

def fig_scree():
    rng = np.random.default_rng(0)
    eig = np.sort(rng.gamma(2.0, 1.0, size=20))[::-1]
    eig[:3] *= 4.0
    evr = eig / eig.sum()
    cevr = np.cumsum(evr)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(1, 21), evr, "o-", color="#1f77b4")
    axes[0].set_title("Scree plot  (individual EVR)")
    axes[0].set_xlabel("component"); axes[0].set_ylabel("explained variance ratio")
    axes[0].axvline(3.5, color="#d62728", ls="--", alpha=0.7, label="elbow")
    axes[0].legend()

    axes[1].plot(np.arange(1, 21), cevr, "s-", color="#ff7f0e")
    for thr in (0.8, 0.9, 0.95):
        k = int(np.searchsorted(cevr, thr) + 1)
        axes[1].axhline(thr, ls="--", color="grey", alpha=0.45)
        axes[1].axvline(k, ls=":", color="grey", alpha=0.45)
        axes[1].text(k + 0.2, thr - 0.04, f"k={k} @ {thr:.0%}", fontsize=9)
    axes[1].set_title("Cumulative EVR — how to pick k")
    axes[1].set_xlabel("number of components"); axes[1].set_ylabel("cumulative EVR")
    fig.tight_layout(); fig.savefig(OUT / "04_scree_and_cumulative.png")
    plt.close(fig)


# ---------- 5. Standardization matters ----------

def fig_standardization():
    rng = np.random.default_rng(3)
    n = 400
    age = rng.normal(40, 12, n)                      # years
    income = rng.normal(60_000, 25_000, n) + age * 500  # dollars, correlated
    X = np.column_stack([age, income])

    pca_raw = PCA(n_components=2).fit(X)
    X_std = StandardScaler().fit_transform(X)
    pca_std = PCA(n_components=2).fit(X_std)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, p, title in [
        (axes[0], X, pca_raw, "Without standardisation\nPC1 ≈ income axis (huge variance)"),
        (axes[1], X_std, pca_std, "With standardisation\nPC1 reflects joint structure"),
    ]:
        ax.scatter(data[:, 0], data[:, 1], s=12, alpha=0.45,
                   color="#4c78a8", edgecolor="white", lw=0.3)
        ctr = p.mean_
        s1 = np.sqrt(p.explained_variance_[0])
        s2 = np.sqrt(p.explained_variance_[1])
        draw_vec(ax, ctr, p.components_[0], 2 * s1, "#d62728", "PC1")
        draw_vec(ax, ctr, p.components_[1], 2 * s2, "#2ca02c", "PC2")
        ax.set_title(title)
    axes[0].set_xlabel("age (yrs)"); axes[0].set_ylabel("income ($)")
    axes[1].set_xlabel("age (z)"); axes[1].set_ylabel("income (z)")
    fig.tight_layout(); fig.savefig(OUT / "05_standardization_effect.png")
    plt.close(fig)


# ---------- 6. Outlier sensitivity ----------

def fig_outlier():
    X = correlated_cloud(n=200, angle_deg=25, sx=2.5, sy=0.8, seed=5)
    X_out = np.vstack([X, [[0, 15]]])  # single extreme outlier

    p1 = PCA(n_components=2).fit(X)
    p2 = PCA(n_components=2).fit(X_out)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, p, title in [
        (axes[0], X, p1, "Clean data — PC1 follows the cloud"),
        (axes[1], X_out, p2, "One outlier → PC1 swings toward it"),
    ]:
        ax.scatter(data[:, 0], data[:, 1], s=18, alpha=0.6,
                   color="#4c78a8", edgecolor="white", lw=0.3)
        ctr = p.mean_
        s1 = np.sqrt(p.explained_variance_[0])
        draw_vec(ax, ctr, p.components_[0], 2.5 * s1, "#d62728", "PC1")
        ax.set_aspect("equal"); ax.set_title(title)
        ax.set_xlim(-7, 7); ax.set_ylim(-7, 17)
    axes[1].annotate("outlier", xy=(0, 15), xytext=(2, 13),
                     arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "06_outlier_sensitivity.png")
    plt.close(fig)


# ---------- 7. Kernel PCA on concentric circles ----------

def fig_kernel_pca():
    X, y = make_circles(n_samples=500, factor=0.4, noise=0.04, random_state=0)
    lin = PCA(n_components=2).fit_transform(X)
    kpc = KernelPCA(n_components=2, kernel="rbf", gamma=12).fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, data, title in [
        (axes[0], X, "Original — non-linear structure"),
        (axes[1], lin, "Linear PCA — still inseparable"),
        (axes[2], kpc, "Kernel PCA (RBF) — unfolds the rings"),
    ]:
        ax.scatter(data[:, 0], data[:, 1], c=y, s=14, cmap="coolwarm",
                   edgecolor="white", lw=0.3, alpha=0.85)
        ax.set_title(title); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(OUT / "07_kernel_pca.png"); plt.close(fig)


# ---------- 8. PCA vs LDA direction ----------

def fig_pca_vs_lda():
    X, y = make_blobs(n_samples=400, centers=[[-2, 0], [2, 0]],
                      cluster_std=[[3.5, 0.6], [3.5, 0.6]], random_state=1)
    t = np.deg2rad(25)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    X = X @ R.T

    p = PCA(n_components=2).fit(X)
    l = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
    w_lda = l.coef_[0] / np.linalg.norm(l.coef_[0])

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=16, alpha=0.55,
               color="#4c78a8", label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, alpha=0.55,
               color="#e45756", label="class 1")
    ctr = X.mean(axis=0)
    draw_vec(ax, ctr, p.components_[0], 4, "#d62728",
             "PCA PC1  (max variance)")
    draw_vec(ax, ctr, w_lda, 4, "#2ca02c",
             "LDA axis  (max class separation)")
    ax.set_aspect("equal"); ax.set_title("PCA is unsupervised; LDA uses labels")
    ax.legend(loc="upper left"); ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    fig.tight_layout(); fig.savefig(OUT / "08_pca_vs_lda.png"); plt.close(fig)


# ---------- 9. SVD block diagram ----------

def fig_svd_diagram():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.15, 0.22, 0.70, "X (centered)\nn × d", "#cfe2f3"),
        (0.30, 0.30, 0.10, 0.45, "U\nn × r", "#d9ead3"),
        (0.44, 0.42, 0.10, 0.18, "Σ\nr × r", "#fce5cd"),
        (0.58, 0.30, 0.22, 0.45, "Vᵀ\nr × d", "#f4cccc"),
    ]
    for x, y, w, h, lbl, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c,
                                   edgecolor="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, lbl, ha="center", va="center",
                fontsize=12, fontweight="bold")
    ax.text(0.265, 0.5, "=", fontsize=22, ha="center", va="center")
    ax.text(0.415, 0.5, "·", fontsize=28, ha="center", va="center")
    ax.text(0.555, 0.5, "·", fontsize=28, ha="center", va="center")
    ax.text(0.5, 0.05,
            "Columns of V = PCA loadings    •    σᵢ² / (n−1) = eigenvalues    •    Scores Z = U Σ",
            ha="center", fontsize=11, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("SVD view of PCA", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "09_svd_diagram.png"); plt.close(fig)


# ---------- 10. Digit reconstruction at increasing k ----------

def fig_reconstruction_k():
    digits = load_digits()
    X = digits.data.astype(float)
    X_c = X - X.mean(axis=0)
    full = PCA(n_components=64).fit(X_c)

    ks = [1, 2, 5, 10, 20, 40, 64]
    sample_idx = [0, 7, 11]  # three sample digits
    fig, axes = plt.subplots(len(sample_idx), len(ks) + 1,
                             figsize=(1.4 * (len(ks) + 1), 1.6 * len(sample_idx)))
    for row, si in enumerate(sample_idx):
        axes[row, 0].imshow(X[si].reshape(8, 8), cmap="gray")
        axes[row, 0].set_title("original" if row == 0 else "", fontsize=10)
        axes[row, 0].set_ylabel(f"digit {digits.target[si]}", fontsize=10)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        for col, k in enumerate(ks, start=1):
            Vk = full.components_[:k]
            z = X_c[si] @ Vk.T
            rec = z @ Vk + X.mean(axis=0)
            axes[row, col].imshow(rec.reshape(8, 8), cmap="gray")
            if row == 0:
                ev = full.explained_variance_ratio_[:k].sum()
                axes[row, col].set_title(f"k={k}\n{ev:.0%}", fontsize=9)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.suptitle("Image reconstruction from top-k components (sklearn digits)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "10_reconstruction_k.png"); plt.close(fig)


if __name__ == "__main__":
    fig_intuition()
    gif_variance_sweep()
    fig_projection_reconstruction()
    fig_scree()
    fig_standardization()
    fig_outlier()
    fig_kernel_pca()
    fig_pca_vs_lda()
    fig_svd_diagram()
    fig_reconstruction_k()
    print("assets written to", OUT)
    for f in sorted(OUT.iterdir()):
        print(" ", f.name, f"{f.stat().st_size // 1024} KB")
