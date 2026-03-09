import gradio as gr

THEME = gr.themes.Base(
    primary_hue="violet",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

BACK_HTML = """
<div style="padding:10px 0 2px;font-family:monospace;">
  <a href="/" style="color:#a78bfa;text-decoration:none;font-size:13px;
                     display:inline-flex;align-items:center;gap:4px;opacity:0.85;"
     onmouseover="this.style.opacity='1'" onmouseout="this.style.opacity='0.85'">
    ← ML Playground
  </a>
</div>
"""
