"""
Clustering — compare K-Means, DBSCAN, GMM, and Agglomerative on 2D datasets.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gradio as gr
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from pages.shared import THEME, BACK_HTML

DATASETS   = ["Blobs", "Moons", "Circles", "Anisotropic", "Varied Density"]
ALGORITHMS = ["K-Means", "DBSCAN", "Gaussian Mixture", "Agglomerative"]

# 20 visually distinct colours for clusters
PALETTE = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e91e63","#00bcd4","#8bc34a","#ff5722",
    "#607d8b","#795548","#ff9800","#673ab7","#03a9f4",
    "#4caf50","#f44336","#9c27b0","#ffeb3b","#009688",
]


def make_dataset(name, n_points, noise, seed=42):
    rng = np.random.RandomState(seed)
    if name == "Blobs":
        X, _ = make_blobs(n_samples=n_points, centers=4, cluster_std=0.85, random_state=seed)
    elif name == "Moons":
        X, _ = make_moons(n_samples=n_points, noise=noise, random_state=seed)
    elif name == "Circles":
        X, _ = make_circles(n_samples=n_points, noise=noise, factor=0.48, random_state=seed)
    elif name == "Anisotropic":
        X, _ = make_blobs(n_samples=n_points, centers=3, random_state=seed)
        X = X @ np.array([[0.65, -0.55], [-0.4, 0.85]])
    else:  # Varied Density
        parts = [
            make_blobs(n_samples=n_points // 4, centers=[[-5, -5]], cluster_std=0.5, random_state=seed)[0],
            make_blobs(n_samples=n_points // 2, centers=[[0,  0]], cluster_std=1.6, random_state=seed)[0],
            make_blobs(n_samples=n_points // 4, centers=[[5,  5]], cluster_std=0.7, random_state=seed)[0],
        ]
        X = np.vstack(parts)
    return StandardScaler().fit_transform(X)


def run_clustering(dataset, algorithm, n_clusters, eps, min_samples, n_points, noise):
    X = make_dataset(dataset, int(n_points), noise)
    n_clusters = int(n_clusters)

    if algorithm == "K-Means":
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    elif algorithm == "DBSCAN":
        labels = DBSCAN(eps=eps, min_samples=int(min_samples)).fit_predict(X)
    elif algorithm == "Gaussian Mixture":
        labels = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(X)
    else:
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)

    unique = np.unique(labels)
    n_found  = int(np.sum(unique >= 0))
    n_noise  = int(np.sum(labels == -1))

    fig = go.Figure()
    for lbl in unique:
        mask  = labels == lbl
        color = "#555555" if lbl == -1 else PALETTE[lbl % len(PALETTE)]
        name  = "Noise" if lbl == -1 else f"Cluster {lbl}"
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers",
            marker=dict(color=color, size=6, opacity=0.85,
                        line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
            name=name,
        ))

    # Centroids (K-Means / GMM)
    if algorithm in ("K-Means", "Gaussian Mixture"):
        from sklearn.cluster import KMeans as _KM
        from sklearn.mixture import GaussianMixture as _GM
        if algorithm == "K-Means":
            centers = _KM(n_clusters=n_clusters, random_state=42, n_init=10).fit(X).cluster_centers_
        else:
            centers = _GM(n_components=n_clusters, random_state=42).fit(X).means_
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1], mode="markers",
            marker=dict(color="white", size=14, symbol="x",
                        line=dict(color="black", width=2)),
            name="Centroids",
        ))

    subtitle = f"{n_found} cluster{'s' if n_found != 1 else ''}"
    if n_noise:
        subtitle += f", {n_noise} noise pts"

    fig.update_layout(
        title=f"{algorithm} on {dataset} — {subtitle}",
        paper_bgcolor="#111111", plot_bgcolor="#111111",
        font=dict(color="white", family="monospace"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False, scaleanchor="x"),
        legend=dict(bgcolor="rgba(40,40,40,0.85)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
        height=540, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def on_alg_change(alg):
    show_k    = alg in ("K-Means", "Gaussian Mixture", "Agglomerative")
    show_dbs  = alg == "DBSCAN"
    return gr.update(visible=show_k), gr.update(visible=show_dbs), gr.update(visible=show_dbs)


with gr.Blocks(title="Clustering — ML Playground") as demo:
    gr.HTML(BACK_HTML)
    gr.Markdown("# Clustering\n### Compare algorithms on 2D data distributions")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            dataset_dd   = gr.Dropdown(choices=DATASETS,   value="Blobs",   label="Dataset")
            algorithm_dd = gr.Dropdown(choices=ALGORITHMS, value="K-Means", label="Algorithm")
            n_points_sl  = gr.Slider(100, 600, step=50,  value=300, label="Points")
            noise_sl     = gr.Slider(0.0, 0.5, step=0.02, value=0.08, label="Noise")
            k_sl         = gr.Slider(2, 10, step=1, value=4, label="Clusters (k)", visible=True)
            eps_sl       = gr.Slider(0.05, 1.0, step=0.02, value=0.3, label="DBSCAN: ε", visible=False)
            min_sl       = gr.Slider(2, 20, step=1, value=5, label="DBSCAN: min_samples", visible=False)
            run_btn      = gr.Button("▶ Run", variant="primary", size="lg")

        with gr.Column(scale=2):
            plot_out = gr.Plot()

    algorithm_dd.change(fn=on_alg_change, inputs=[algorithm_dd], outputs=[k_sl, eps_sl, min_sl])
    run_btn.click(
        fn=run_clustering,
        inputs=[dataset_dd, algorithm_dd, k_sl, eps_sl, min_sl, n_points_sl, noise_sl],
        outputs=[plot_out],
    )
