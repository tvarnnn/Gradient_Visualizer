"""
Decision Boundaries — show how classifiers carve up 2D feature space.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gradio as gr
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pages.shared import THEME, BACK_HTML

DATASETS    = ["Moons", "Circles", "Blobs (3-class)", "XOR"]
CLASSIFIERS = ["KNN", "SVM (RBF)", "SVM (Linear)", "Decision Tree", "Random Forest", "Logistic Regression"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]


def make_dataset(name, n_points, noise, seed=42):
    rng = np.random.RandomState(seed)
    if name == "Moons":
        X, y = make_moons(n_samples=n_points, noise=noise, random_state=seed)
    elif name == "Circles":
        X, y = make_circles(n_samples=n_points, noise=noise, factor=0.45, random_state=seed)
    elif name == "Blobs (3-class)":
        X, y = make_blobs(n_samples=n_points, centers=3, cluster_std=max(0.3, noise * 6), random_state=seed)
    else:  # XOR
        X = rng.randn(n_points, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X += rng.randn(*X.shape) * noise
    return X, y


def get_clf(name, k, C, depth):
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=k)
    elif name == "SVM (RBF)":
        return SVC(kernel="rbf", C=C, probability=True)
    elif name == "SVM (Linear)":
        return SVC(kernel="linear", C=C, probability=True)
    elif name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=depth, random_state=42)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=80, max_depth=depth, random_state=42)
    else:
        return LogisticRegression(C=C, max_iter=500, random_state=42)


def run_decision(dataset, clf_name, n_points, noise, k, C, depth):
    X, y = make_dataset(dataset, n_points, noise)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = get_clf(clf_name, int(k), C, int(depth))
    clf.fit(Xs, y)
    score = clf.score(Xs, y)

    # Mesh for boundary
    h = 0.025
    x1, x2 = Xs[:, 0].min() - 0.6, Xs[:, 0].max() + 0.6
    y1, y2 = Xs[:, 1].min() - 0.6, Xs[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x1, x2, h), np.arange(y1, y2, h))
    grid   = np.c_[xx.ravel(), yy.ravel()]

    n_classes = len(np.unique(y))
    if hasattr(clf, "predict_proba") and n_classes == 2:
        Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
        colorscale = [[0, "rgba(231,76,60,0.35)"], [0.5, "rgba(150,150,150,0.05)"], [1, "rgba(52,152,219,0.35)"]]
        zmin, zmax = 0.0, 1.0
    else:
        Z = clf.predict(grid).astype(float).reshape(xx.shape)
        colorscale = [
            [0.0, "rgba(231,76,60,0.35)"],
            [0.5, "rgba(52,152,219,0.35)"],
            [1.0, "rgba(46,204,113,0.35)"],
        ]
        zmin, zmax = 0, n_classes - 1

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=np.arange(x1, x2, h), y=np.arange(y1, y2, h), z=Z,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=False, hoverinfo="skip",
    ))

    for cls_idx in range(n_classes):
        mask = y == cls_idx
        color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
        label = f"Class {cls_idx}"
        fig.add_trace(go.Scatter(
            x=Xs[mask, 0], y=Xs[mask, 1], mode="markers",
            marker=dict(color=color, size=8, line=dict(color="white", width=0.8), opacity=0.9),
            name=label,
        ))

    fig.update_layout(
        title=f"{clf_name} on {dataset} — Train Accuracy: {score:.1%}",
        paper_bgcolor="#111111", plot_bgcolor="#111111",
        font=dict(color="white", family="monospace"),
        xaxis=dict(title="Feature 1", gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(title="Feature 2", gridcolor="rgba(255,255,255,0.08)", zeroline=False, scaleanchor="x"),
        legend=dict(bgcolor="rgba(40,40,40,0.85)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
        height=540, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def on_clf_change(name):
    return (
        gr.update(visible=(name == "KNN")),
        gr.update(visible=(name in ["SVM (RBF)", "SVM (Linear)", "Logistic Regression"])),
        gr.update(visible=(name in ["Decision Tree", "Random Forest"])),
    )


with gr.Blocks(title="Decision Boundaries — ML Playground") as demo:
    gr.HTML(BACK_HTML)
    gr.Markdown("# Decision Boundaries\n### See how classifiers carve up 2D feature space")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            dataset_dd  = gr.Dropdown(choices=DATASETS, value="Moons", label="Dataset")
            clf_dd      = gr.Dropdown(choices=CLASSIFIERS, value="KNN", label="Classifier")
            n_points_sl = gr.Slider(100, 500, step=50, value=250, label="Points")
            noise_sl    = gr.Slider(0.0, 0.45, step=0.02, value=0.15, label="Noise")
            k_sl        = gr.Slider(1, 25, step=1, value=5, label="K (neighbors)", visible=True)
            c_sl        = gr.Slider(0.01, 20.0, step=0.1, value=1.0, label="Regularization C", visible=False)
            depth_sl    = gr.Slider(1, 15, step=1, value=5, label="Max Depth", visible=False)
            run_btn     = gr.Button("▶ Run", variant="primary", size="lg")

        with gr.Column(scale=2):
            plot_out = gr.Plot()

    clf_dd.change(fn=on_clf_change, inputs=[clf_dd], outputs=[k_sl, c_sl, depth_sl])
    run_btn.click(
        fn=run_decision,
        inputs=[dataset_dd, clf_dd, n_points_sl, noise_sl, k_sl, c_sl, depth_sl],
        outputs=[plot_out],
    )
