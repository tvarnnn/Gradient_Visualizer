"""
ML Playground — FastAPI + Gradio multi-page app.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 7860 --reload

Landing page is served at /, each visualization is mounted at its own path.
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import gradio as gr

from pages.gradient_page    import demo as gradient_demo
from pages.decision_page    import demo as decision_demo
from pages.overfitting_page import demo as overfitting_demo
from pages.clustering_page  import demo as clustering_demo
from pages.momentum_page    import demo as momentum_demo
from pages.nn_trainer_page  import demo as nn_demo
from pages.rl_page          import demo as rl_demo
from pages.shared           import THEME

app = FastAPI(title="ML Playground")

# ── Landing page ──────────────────────────────────────────────────────────────

_LANDING = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ML Playground</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
    body {
      background:#0d0d1a;
      color:#e2e8f0;
      font-family:'Inter',system-ui,-apple-system,sans-serif;
      min-height:100vh;
    }
    body::before {
      content:'';
      position:fixed; inset:0;
      background-image:
        linear-gradient(rgba(124,58,237,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,0.04) 1px, transparent 1px);
      background-size:52px 52px;
      pointer-events:none; z-index:0;
    }
    .wrap { position:relative; z-index:1; }

    header { text-align:center; padding:72px 20px 48px; }

    .badge {
      display:inline-block;
      padding:4px 16px;
      background:rgba(124,58,237,0.15);
      border:1px solid rgba(124,58,237,0.4);
      border-radius:999px;
      color:#a78bfa;
      font-size:0.78rem; font-weight:600;
      letter-spacing:0.1em; text-transform:uppercase;
      margin-bottom:22px;
    }

    h1 {
      font-size:clamp(2.4rem,6vw,4rem);
      font-weight:800; letter-spacing:-0.03em; line-height:1.1;
      background:linear-gradient(135deg,#c4b5fd 0%,#60a5fa 50%,#34d399 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      background-clip:text; margin-bottom:16px;
    }

    .sub {
      color:#94a3b8; font-size:1.05rem;
      max-width:480px; margin:0 auto; line-height:1.65;
    }

    .grid {
      display:grid;
      grid-template-columns:repeat(auto-fill,minmax(290px,1fr));
      gap:18px;
      max-width:1080px; margin:0 auto; padding:0 24px 80px;
    }

    a.card {
      background:rgba(26,26,46,0.82);
      border:1px solid rgba(167,139,250,0.14);
      border-radius:16px; padding:28px 24px;
      cursor:pointer; text-decoration:none; color:inherit;
      display:flex; flex-direction:column; gap:10px;
      transition:transform .2s,border-color .2s,box-shadow .2s,background .2s;
      backdrop-filter:blur(6px);
    }
    a.card:hover {
      transform:translateY(-5px);
      border-color:rgba(167,139,250,0.5);
      box-shadow:0 14px 44px rgba(124,58,237,0.22),0 0 0 1px rgba(167,139,250,0.1);
      background:rgba(30,24,54,0.92);
    }
    a.card:active { transform:translateY(-2px); }

    .icon  { font-size:1.8rem; line-height:1; }
    .title { font-size:1.1rem; font-weight:700; color:#f1f5f9; margin-top:4px; }
    .desc  { font-size:0.875rem; color:#94a3b8; line-height:1.55; flex:1; }

    .foot-row {
      display:flex; align-items:center;
      justify-content:space-between; margin-top:6px;
    }

    .live {
      display:inline-flex; align-items:center; gap:5px;
      padding:3px 10px; border-radius:999px;
      font-size:0.72rem; font-weight:600;
      background:rgba(52,211,153,0.1);
      color:#34d399; border:1px solid rgba(52,211,153,0.25);
    }
    .live::before {
      content:''; width:6px; height:6px; border-radius:50%;
      background:#34d399; animation:pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

    .arr { color:#475569; font-size:1rem; transition:transform .2s,color .2s; }
    a.card:hover .arr { transform:translateX(4px); color:#a78bfa; }

    footer { text-align:center; padding:0 20px 32px; color:#334155; font-size:.8rem; }
    footer a { color:#475569; text-decoration:none; }
    footer a:hover { color:#a78bfa; }
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="badge">Interactive ML Visualizer</div>
    <h1>ML Playground</h1>
    <p class="sub">Explore machine learning concepts through real-time,
    interactive visualizations. Pick a module to dive in.</p>
  </header>

  <div class="grid">

    <a href="/gradient" class="card">
      <div class="icon"></div>
      <div class="title">Loss Landscape</div>
      <div class="desc">Watch SGD, Adam, RMSProp and others navigate 3D loss
      surfaces. Rotate, zoom, scrub, and compare optimizer paths side by side.</div>
      <div class="foot-row"><span class="live">Live</span><span class="arr">→</span></div>
    </a>

    <a href="/decision" class="card">
      <div class="icon">️</div>
      <div class="title">Decision Boundaries</div>
      <div class="desc">See how KNN, SVM, Decision Trees, and Random Forests
      carve up 2D feature space on moons, circles, XOR, and blob datasets.</div>
      <div class="foot-row"><span class="live">Live</span><span class="arr">→</span></div>
    </a>

    <a href="/overfitting" class="card">
      <div class="icon"></div>
      <div class="title">Overfitting vs Generalization</div>
      <div class="desc">Fit polynomials of increasing degree to noisy data.
      Watch training error drop while test error rises — and see regularization fix it.</div>
      <div class="foot-row"><span class="live">Live</span><span class="arr">→</span></div>
    </a>

    <a href="/clustering" class="card">
      <div class="icon"></div>
      <div class="title">Clustering</div>
      <div class="desc">Compare K-Means, DBSCAN, Gaussian Mixture, and
      Agglomerative clustering on moons, circles, blobs, and anisotropic distributions.</div>
      <div class="foot-row"><span class="live">Live</span><span class="arr">→</span></div>
    </a>

    <a href="/momentum" class="card">
      <div class="icon">⚡</div>
      <div class="title">Momentum Dynamics</div>
      <div class="desc">See how momentum, adaptive rates, and gradient
      accumulation change the optimization path on a 1D loss surface in real time.</div>
      <div class="foot-row"><span class="live">Live</span><span class="arr">→</span></div>
    </a>

    <a href="/nn-trainer" class="card">
      <div class="icon"></div>
      <div class="title">Neural Network Trainer</div>
      <div class="desc">Train a live MLP with numpy backprop and watch the
      decision boundary evolve epoch by epoch alongside the training loss curve.</div>
      <div class="foot-row"><span class="live">Live</span><span class="arr">→</span></div>
    </a>

    <a href="/rl" class="card">
      <div class="icon"></div>
      <div class="title">RL: Value Iteration</div>
      <div class="desc">Watch value iteration converge on a grid world, with
      live value function heatmap and greedy policy arrows updating each sweep.</div>
      <div class="foot-row"><span class="live">Live</span><span class="arr">→</span></div>
    </a>

  </div>

  <footer>
    ML Playground &mdash; built with
    <a href="https://gradio.app">Gradio</a> + <a href="https://fastapi.tiangolo.com">FastAPI</a>
  </footer>
</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return _LANDING


# ── Mount each Gradio app ─────────────────────────────────────────────────────
app = gr.mount_gradio_app(app, gradient_demo,    path="/gradient",    theme=THEME)
app = gr.mount_gradio_app(app, decision_demo,    path="/decision",    theme=THEME)
app = gr.mount_gradio_app(app, overfitting_demo, path="/overfitting", theme=THEME)
app = gr.mount_gradio_app(app, clustering_demo,  path="/clustering",  theme=THEME)
app = gr.mount_gradio_app(app, momentum_demo,    path="/momentum",    theme=THEME)
app = gr.mount_gradio_app(app, nn_demo,          path="/nn-trainer",  theme=THEME)
app = gr.mount_gradio_app(app, rl_demo,          path="/rl",          theme=THEME)
