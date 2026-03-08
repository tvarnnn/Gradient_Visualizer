"""
Gradio UI for Gradience.

Animation is now entirely client-side:
  • Python computes the optimizer paths, serialises them to JSON, and
    returns a self-contained HTML string (via build_animation_html).
  • The browser receives that HTML once, and all play/pause/scrub/speed
    controls are pure JavaScript — zero server round-trips.

Benefits over the old generator approach:
  • Pause is instantaneous (JS setInterval / clearInterval)
  • Camera is truly preserved — Plotly.react() is called directly from JS
    with uirevision:'camera', so the camera state is never reset
  • Speed changes take effect immediately mid-playback
  • A scrub slider lets the user jump to any step freely
"""

import gradio as gr

from surfaces import PRESET_SURFACES, make_custom_surface
from visualization import compute_paths, build_surface_mesh, build_animation_html
from optimizers import OPTIMIZERS

SURFACE_NAMES   = list(PRESET_SURFACES.keys()) + ["Custom"]
OPTIMIZER_NAMES = list(OPTIMIZERS.keys())
SPEED_OPTIONS   = ["Slow", "Medium", "Fast", "Lightning"]


# Main callback — runs once per click, returns HTML
def run_visualization(
    surface_name,
    custom_expr,
    selected_optimizers,
    lr,
    x0,
    y0,
    n_steps,
    speed,
):
    """
    Compute all optimizer paths upfront, then return a self-contained
    HTML animation. Two yields: a loading message while computing,
    then the final interactive HTML.
    """
    if not selected_optimizers:
        yield "<p style='color:#f87171;font-family:monospace'>⚠️ Please select at least one optimizer.</p>", ""
        return

    try:
        # Show a loading message while Python works
        yield "<p style='color:#a78bfa;font-family:monospace;padding:12px'>⏳ Computing paths…</p>", "⏳ Computing…"

        # Resolve surface
        if surface_name == "Custom":
            if not custom_expr.strip():
                yield "<p style='color:#f87171;font-family:monospace'>⚠️ Please enter a custom function expression.</p>", ""
                return
            surface = make_custom_surface(custom_expr.strip())
        else:
            surface = PRESET_SURFACES[surface_name]

        # Compute mesh and paths (the only Python work)
        x_lin, y_lin, X, Y, Z = build_surface_mesh(surface)
        paths, converged_at    = compute_paths(surface, selected_optimizers, lr, x0, y0, int(n_steps))

        # Build self-contained HTML — all animation logic is inside
        html = build_animation_html(surface, x_lin, y_lin, X, Y, Z, paths, converged_at, speed)

        # Build a concise summary for the status bar
        names      = ", ".join(selected_optimizers)
        max_steps  = max(len(p[0]) for p in paths.values())
        conv_notes = [f"{n} @ step {s}" for n, s in converged_at.items()]
        conv_str   = ("  |  Converged: " + ", ".join(conv_notes)) if conv_notes else ""
        status     = f"Ready — **{names}** on **{surface.name}** ({max_steps} steps){conv_str}"

        yield html, status

    except Exception as exc:
        yield f"<p style='color:#f87171;font-family:monospace'> Error: {exc}</p>", f" {exc}"


# Surface-change callback

def on_surface_change(surface_name):
    """Reset start sliders and show/hide the custom expression box."""
    if surface_name == "Custom":
        return gr.update(visible=True), -2.0, 2.0
    surface = PRESET_SURFACES[surface_name]
    x0, y0  = surface.default_start
    return gr.update(visible=False), x0, y0


# Gradio layout
with gr.Blocks(title="Gradience — ML Optimizer Visualization") as demo:

    gr.Markdown(
        """
        # Gradience
        ### Watch ML Optimizers Navigate Loss Landscapes in Real Time
        Configure your run on the left, hit **Run**, then use the **Play / Pause / Scrub**
        controls inside the animation panel. The 3-D camera is freeform — rotate and
        zoom freely and it will stay locked between steps.
        """
    )

    with gr.Row():

        # LEFT — all input controls
        with gr.Column(scale=1, min_width=300):

            gr.Markdown("### 🗺️ Loss Surface")
            surface_dropdown = gr.Dropdown(
                choices=SURFACE_NAMES,
                value="Rosenbrock (Banana)",
                label="Preset Surface",
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
                choices=OPTIMIZER_NAMES,
                value=["SGD", "Adam"],
                label="Select optimizers (pick multiple to compare)",
            )
            lr_slider = gr.Slider(
                minimum=0.001, maximum=0.5, step=0.001, value=0.01,
                label="Learning Rate",
                info="Shared across all selected optimizers",
            )

            gr.Markdown("### Starting Point")
            x0_slider = gr.Slider(minimum=-5.0, maximum=5.0, step=0.1, value=-1.5, label="x₀")
            y0_slider = gr.Slider(minimum=-5.0, maximum=5.0, step=0.1, value=2.0,  label="y₀")

            gr.Markdown("### Animation")
            steps_slider = gr.Slider(minimum=10, maximum=500, step=10, value=100, label="Number of Steps")
            speed_dropdown = gr.Dropdown(
                choices=SPEED_OPTIONS, value="Medium",
                label="Initial Playback Speed",
                info="Can also be changed inside the animation panel",
            )

            run_btn     = gr.Button("▶  Run", variant="primary", size="lg")
            status_text = gr.Markdown("")

        # RIGHT — the self-contained animation HTML
        # Play / Pause / Stop / Scrub are all inside the HTML panel
        with gr.Column(scale=2):
            anim_output = gr.HTML(
                value="<p style='color:#888;font-family:monospace;padding:20px'>"
                      "Configure your run and click <b>▶ Run</b> to start.</p>"
            )

    # ---- Reference tables ----
    with gr.Accordion("📚 Optimizer Reference", open=False):
        gr.Markdown(
            """
| Optimizer | Key Idea | Best For |
|-----------|----------|----------|
| **SGD** | Subtract lr × gradient at every step — nothing more | Simple baseline, convex problems |
| **Momentum** | Accumulates a velocity vector to smooth oscillations | Ravines, noisy gradients |
| **AdaGrad** | Divides lr by the running sum of squared gradients | Sparse features / embeddings |
| **RMSProp** | Like AdaGrad but uses an EMA so the rate doesn't vanish | Non-stationary / RNN training |
| **Adam** | Momentum + RMSProp + bias correction — the modern default | Almost everything |
"""
        )

    with gr.Accordion(" Surface Reference", open=False):
        gr.Markdown(
            """
| Surface | Known Minimum | Why It's Interesting |
|---------|--------------|----------------------|
| **Bowl** | (0, 0) | Perfectly convex — all optimizers converge easily |
| **Rosenbrock** | (1, 1) | Flat narrow valley — the classic hard benchmark |
| **Himmelblau** | Four minima | Multiple basins — which one does each optimizer find? |
| **Saddle Point** | (0, 0) is a saddle | Shows how optimizers escape (or don't) |
| **Beale** | (3, 0.5) | Flat plateaus + sharp walls — tests adaptive methods |
"""
        )

    # Event wiring

    surface_dropdown.change(
        fn=on_surface_change,
        inputs=[surface_dropdown],
        outputs=[custom_expr, x0_slider, y0_slider],
    )

    run_btn.click(
        fn=run_visualization,
        inputs=[
            surface_dropdown, custom_expr, optimizer_checkboxes,
            lr_slider, x0_slider, y0_slider, steps_slider, speed_dropdown,
        ],
        outputs=[anim_output, status_text],
    )


# Local launch

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(
            primary_hue="violet",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        )
    )
