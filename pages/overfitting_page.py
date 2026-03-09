"""
Overfitting vs Generalization — polynomial regression with train/test error curves.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pages.shared import THEME, BACK_HTML

MAX_DEGREE = 14


def fit_poly(X_train, y_train, degree, lam):
    """Ridge regression via normal equations: (XᵀX + λI)⁻¹Xᵀy."""
    Phi = np.vander(X_train, degree + 1, increasing=True)
    reg = lam * np.eye(degree + 1)
    reg[0, 0] = 0          # don't regularize bias
    w = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y_train)
    return w


def predict_poly(X, w):
    degree = len(w) - 1
    Phi = np.vander(X, degree + 1, increasing=True)
    return Phi @ w


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def run_overfitting(n_points, noise, lam, function_name, test_frac):
    rng = np.random.RandomState(42)
    x_all = np.sort(rng.uniform(-1, 1, n_points))

    if function_name == "sin(2πx)":
        y_true = np.sin(2 * np.pi * x_all)
    elif function_name == "x³ − x":
        y_true = x_all ** 3 - x_all
    elif function_name == "exp(−x²)·cos(4x)":
        y_true = np.exp(-(x_all ** 2)) * np.cos(4 * x_all)
    else:
        y_true = 0.5 * x_all ** 2 + 0.3 * x_all

    y_all = y_true + rng.randn(n_points) * noise

    n_test  = max(5, int(n_points * test_frac))
    n_train = n_points - n_test
    idx     = rng.permutation(n_points)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_test,  y_test  = x_all[test_idx],  y_all[test_idx]

    x_curve = np.linspace(-1, 1, 400)
    degrees = list(range(1, MAX_DEGREE + 1))
    train_mses, test_mses = [], []

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=["Fitted Curve", "Train vs Test Error"],
    )

    COLORS = [
        "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c",
        "#3498db", "#9b59b6", "#e91e63", "#00bcd4", "#8bc34a",
        "#ff5722", "#607d8b", "#795548", "#9c27b0",
    ]

    # Plot data points (left panel)
    fig.add_trace(go.Scatter(
        x=x_train, y=y_train, mode="markers",
        marker=dict(color="#60a5fa", size=6, opacity=0.7),
        name="Train", legendgroup="data",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_test, y=y_test, mode="markers",
        marker=dict(color="#f87171", size=6, symbol="x", opacity=0.8),
        name="Test", legendgroup="data",
    ), row=1, col=1)

    # True function
    if function_name == "sin(2πx)":
        y_curve_true = np.sin(2 * np.pi * x_curve)
    elif function_name == "x³ − x":
        y_curve_true = x_curve ** 3 - x_curve
    elif function_name == "exp(−x²)·cos(4x)":
        y_curve_true = np.exp(-(x_curve ** 2)) * np.cos(4 * x_curve)
    else:
        y_curve_true = 0.5 * x_curve ** 2 + 0.3 * x_curve

    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve_true, mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"),
        name="True f(x)", showlegend=True,
    ), row=1, col=1)

    # Fit each degree
    for deg in degrees:
        w = fit_poly(x_train, y_train, deg, lam)
        y_hat_train = predict_poly(x_train, w)
        y_hat_test  = predict_poly(x_test,  w)
        train_mses.append(mse(y_train, y_hat_train))
        test_mses.append(mse(y_test,  y_hat_test))

    # Highlight 3 fitted curves on the left: degree 1, ~middle, max
    highlight_degrees = sorted(set([1, MAX_DEGREE // 2, MAX_DEGREE]))
    for deg in highlight_degrees:
        w = fit_poly(x_train, y_train, deg, lam)
        y_curve_fit = predict_poly(x_curve, w)
        color = COLORS[deg - 1]
        fig.add_trace(go.Scatter(
            x=x_curve, y=y_curve_fit, mode="lines",
            line=dict(color=color, width=2),
            name=f"deg {deg}",
        ), row=1, col=1)

    # Error curves (right panel)
    fig.add_trace(go.Scatter(
        x=degrees, y=train_mses, mode="lines+markers",
        line=dict(color="#60a5fa", width=2.5),
        marker=dict(size=5), name="Train MSE", legendgroup="err",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=degrees, y=test_mses, mode="lines+markers",
        line=dict(color="#f87171", width=2.5),
        marker=dict(size=5), name="Test MSE", legendgroup="err",
    ), row=1, col=2)

    best_deg = int(np.argmin(test_mses)) + 1
    best_test = test_mses[best_deg - 1]
    fig.add_trace(go.Scatter(
        x=[best_deg], y=[best_test], mode="markers",
        marker=dict(color="#34d399", size=12, symbol="star"),
        name=f"Best test (deg {best_deg})", legendgroup="err",
    ), row=1, col=2)

    _G = "rgba(255,255,255,0.08)"
    fig.update_layout(
        paper_bgcolor="#111111", plot_bgcolor="#111111",
        font=dict(color="white", family="monospace"),
        legend=dict(bgcolor="rgba(40,40,40,0.85)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        title=f"Overfitting — best test error at degree {best_deg} (λ={lam:.3f})",
    )
    for axis in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        fig.update_layout(**{axis: dict(gridcolor=_G, zerolinecolor="rgba(255,255,255,0.15)")})
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_xaxes(title_text="Polynomial Degree", row=1, col=2)
    fig.update_yaxes(title_text="MSE", row=1, col=2)

    return fig


FUNCTIONS = ["sin(2πx)", "x³ − x", "exp(−x²)·cos(4x)", "Quadratic"]

with gr.Blocks(title="Overfitting — ML Playground") as demo:
    gr.HTML(BACK_HTML)
    gr.Markdown(
        "# Overfitting vs Generalization\n"
        "### Fit polynomials of increasing degree — watch train error drop while test error rises"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            fn_dd       = gr.Dropdown(choices=FUNCTIONS, value="sin(2πx)", label="True Function")
            n_pts_sl    = gr.Slider(30, 200, step=10, value=60, label="Total Points")
            noise_sl    = gr.Slider(0.0, 0.8, step=0.02, value=0.2, label="Noise σ")
            test_sl     = gr.Slider(0.1, 0.5, step=0.05, value=0.3, label="Test Fraction")
            lam_sl      = gr.Slider(0.0, 1.0, step=0.001, value=0.0,
                                    label="Ridge Regularization λ",
                                    info="λ=0 → no regularization")
            run_btn     = gr.Button("▶ Run", variant="primary", size="lg")

        with gr.Column(scale=2):
            plot_out = gr.Plot()

    run_btn.click(
        fn=run_overfitting,
        inputs=[n_pts_sl, noise_sl, lam_sl, fn_dd, test_sl],
        outputs=[plot_out],
    )
