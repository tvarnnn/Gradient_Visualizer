"""
Gradient / Loss Landscape page — adapted from the original app.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
from surfaces import PRESET_SURFACES, make_custom_surface
from visualization import compute_paths, build_surface_mesh, build_animation_html
from optimizers import OPTIMIZERS
from pages.shared import THEME, BACK_HTML

SURFACE_NAMES   = list(PRESET_SURFACES.keys()) + ["Custom"]
OPTIMIZER_NAMES = list(OPTIMIZERS.keys())
SPEED_OPTIONS   = ["Slow", "Medium", "Fast", "Lightning"]


def run_visualization(surface_name, custom_expr, selected_optimizers,
                      lr, x0, y0, n_steps, speed):
    if not selected_optimizers:
        yield "<p style='color:#f87171;font-family:monospace'>⚠️ Select at least one optimizer.</p>", ""
        return
    try:
        yield "<p style='color:#a78bfa;font-family:monospace;padding:12px'>⏳ Computing paths…</p>", "⏳ Computing…"
        if surface_name == "Custom":
            if not custom_expr.strip():
                yield "<p style='color:#f87171;font-family:monospace'>⚠️ Enter a custom function.</p>", ""
                return
            surface = make_custom_surface(custom_expr.strip())
        else:
            surface = PRESET_SURFACES[surface_name]

        x_lin, y_lin, X, Y, Z = build_surface_mesh(surface)
        paths, converged_at = compute_paths(surface, selected_optimizers, lr, x0, y0, int(n_steps))
        html = build_animation_html(surface, x_lin, y_lin, X, Y, Z, paths, converged_at, speed)

        names     = ", ".join(selected_optimizers)
        max_steps = max(len(p[0]) for p in paths.values())
        conv_notes = [f"{n} @ step {s}" for n, s in converged_at.items()]
        conv_str   = ("  |  Converged: " + ", ".join(conv_notes)) if conv_notes else ""
        status = f"Ready — **{names}** on **{surface.name}** ({max_steps} steps){conv_str}"
        yield html, status
    except Exception as exc:
        yield f"<p style='color:#f87171;font-family:monospace'>Error: {exc}</p>", f"Error: {exc}"


def on_surface_change(surface_name):
    if surface_name == "Custom":
        return gr.update(visible=True), -2.0, 2.0
    surface = PRESET_SURFACES[surface_name]
    x0, y0 = surface.default_start
    return gr.update(visible=False), x0, y0


with gr.Blocks(title="Loss Landscape — ML Playground") as demo:
    gr.HTML(BACK_HTML)
    gr.Markdown(
        """
        # Loss Landscape
        ### Watch ML Optimizers Navigate Loss Surfaces in Real Time
        Configure your run on the left, hit **Run**, then use the **Play / Pause / Scrub**
        controls inside the animation panel. The 3-D camera stays locked between steps.
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🗺️ Loss Surface")
            surface_dropdown = gr.Dropdown(
                choices=SURFACE_NAMES, value="Rosenbrock (Banana)", label="Preset Surface",
                info="Pick a preset landscape or define your own",
            )
            custom_expr = gr.Textbox(
                label="Custom f(x, y)",
                placeholder="e.g.  sin(x) * cos(y) + 0.1*(x**2 + y**2)",
                info="Allowed: x, y, sin, cos, tan, exp, log, sqrt, abs, pi, e",
                visible=False,
            )
            gr.Markdown("### Optimizers")
            optimizer_checkboxes = gr.CheckboxGroup(
                choices=OPTIMIZER_NAMES, value=["SGD", "Adam"],
                label="Select optimizers (pick multiple to compare)",
            )
            lr_slider = gr.Slider(minimum=0.001, maximum=0.5, step=0.001, value=0.01,
                                  label="Learning Rate", info="Shared across all selected optimizers")
            gr.Markdown("### Starting Point")
            x0_slider = gr.Slider(minimum=-5.0, maximum=5.0, step=0.1, value=-1.5, label="x₀")
            y0_slider = gr.Slider(minimum=-5.0, maximum=5.0, step=0.1, value=2.0,  label="y₀")
            gr.Markdown("### Animation")
            steps_slider = gr.Slider(minimum=10, maximum=500, step=10, value=100, label="Number of Steps")
            speed_dropdown = gr.Dropdown(choices=SPEED_OPTIONS, value="Medium",
                                         label="Initial Playback Speed",
                                         info="Can also be changed inside the animation panel")
            run_btn     = gr.Button("▶  Run", variant="primary", size="lg")
            status_text = gr.Markdown("")

        with gr.Column(scale=2):
            anim_output = gr.HTML(
                value="<p style='color:#888;font-family:monospace;padding:20px'>"
                      "Configure your run and click <b>▶ Run</b> to start.</p>"
            )

    with gr.Accordion("📚 Optimizer Reference", open=False):
        gr.Markdown(
            """
| Optimizer | Key Idea | Best For |
|-----------|----------|----------|
| **SGD** | Subtract lr × gradient | Simple baseline, convex problems |
| **Momentum** | Accumulates velocity to smooth oscillations | Ravines, noisy gradients |
| **AdaGrad** | Divides lr by running sum of squared gradients | Sparse features / embeddings |
| **RMSProp** | Like AdaGrad but uses EMA so rate doesn't vanish | Non-stationary / RNN training |
| **Adam** | Momentum + RMSProp + bias correction | Almost everything |
"""
        )

    surface_dropdown.change(
        fn=on_surface_change, inputs=[surface_dropdown],
        outputs=[custom_expr, x0_slider, y0_slider],
    )
    run_btn.click(
        fn=run_visualization,
        inputs=[surface_dropdown, custom_expr, optimizer_checkboxes,
                lr_slider, x0_slider, y0_slider, steps_slider, speed_dropdown],
        outputs=[anim_output, status_text],
    )
