"""
visualization.py
Public API:

  build_surface_mesh(surface)
      Returns (x_lin, y_lin, X, Y, Z).

  compute_paths(surface, optimizers, lr, x0, y0, n_steps)
      Runs each optimizer, records trajectories.
      Returns (paths, converged_at).

  build_animation_html(surface, x_lin, y_lin, X, Y, Z, paths, converged_at)
      Serialises all path data to JSON and returns a self-contained HTML
      string with embedded Plotly + JavaScript that handles all animation
      client-side.

Why client-side?
  The previous approach yielded a new Plotly figure from Python on every
  step. Each yield incurred a full browser→server→browser round-trip, so
  pause/stop had unavoidable lag and Plotly.react() was never called
  directly (Gradio owns the render cycle), meaning uirevision had no
  reliable effect on the camera.

  With the HTML approach:
    • Play/pause/scrub are pure JS — instant, zero latency
    • Plotly.react() is called directly from JS with uirevision:'camera'
      so the camera is truly preserved across every frame update
    • Python is only involved once: to compute paths and emit the HTML
"""

import json
import numpy as np

from surfaces import Surface
from optimizers import OPTIMIZERS, OPTIMIZER_COLORS

RESOLUTION      = 60
CONVERGENCE_TOL = 1e-6

SPEED_DELAYS = {          # kept for reference / future use
    "Slow":      0.45,
    "Medium":    0.15,
    "Fast":      0.04,
    "Lightning": 0.008,
}

SPEED_MS = {              # milliseconds — used in the JS template
    "Slow":      450,
    "Medium":    150,
    "Fast":      40,
    "Lightning": 8,
}


# Surface mesh

def build_surface_mesh(surface: Surface):
    x_lin = np.linspace(*surface.x_range, RESOLUTION)
    y_lin = np.linspace(*surface.y_range, RESOLUTION)
    X, Y  = np.meshgrid(x_lin, y_lin)
    Z     = surface.f(X, Y) if surface.vectorized else np.vectorize(surface.f)(X, Y)
    if surface.z_clip is not None:
        Z = np.clip(Z, None, surface.z_clip)
    return x_lin, y_lin, X, Y, Z


# Optimizer paths

def compute_paths(
    surface: Surface,
    selected_optimizers: list,
    lr: float,
    x0: float,
    y0: float,
    n_steps: int,
):
    """
    Returns (paths, converged_at).
    paths        : { name: (px, py, pz) }
    converged_at : { name: step_index }
    """
    paths:        dict = {}
    converged_at: dict = {}

    for opt_name in selected_optimizers:
        opt    = OPTIMIZERS[opt_name](lr)
        params = np.array([float(x0), float(y0)])
        px, py, pz = [params[0]], [params[1]], [surface.f(*params)]

        for step_idx in range(n_steps):
            grads = surface.gradient(*params)
            if np.any(np.isnan(grads)) or np.linalg.norm(grads) > 1e7:
                break
            if np.linalg.norm(grads) < CONVERGENCE_TOL:
                converged_at[opt_name] = step_idx
                break

            params = opt.step(params, grads)
            params = np.clip(
                params,
                [surface.x_range[0], surface.y_range[0]],
                [surface.x_range[1], surface.y_range[1]],
            )
            z = surface.f(*params)
            if np.isnan(z) or np.isinf(z):
                break
            px.append(params[0]); py.append(params[1]); pz.append(z)

        pz_arr = np.array(pz)
        if surface.z_clip is not None:
            pz_arr = np.clip(pz_arr, None, surface.z_clip)
        paths[opt_name] = (np.array(px), np.array(py), pz_arr)

    return paths, converged_at


# HTML animation builder

# Inline JS/HTML template.
# Placeholders are %%NAME%% style to avoid conflicts with Python f-strings
# and JSON curly braces.
_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>* { box-sizing: border-box; margin: 0; padding: 0; } body { background: #111111; }</style>
</head>
<body>
<div style="background:#111111;color:white;font-family:monospace;border-radius:8px;overflow:hidden;">

  <!-- Transport controls -->
  <div style="padding:8px 12px;display:flex;align-items:center;gap:10px;
              background:#1a1a2e;flex-wrap:wrap;">
    <button id="g_play"  style="padding:5px 14px;background:#7c3aed;color:white;
                                border:none;border-radius:4px;cursor:pointer;font-family:monospace;">
      &#9654; Play
    </button>
    <button id="g_stop"  style="padding:5px 14px;background:#dc2626;color:white;
                                border:none;border-radius:4px;cursor:pointer;font-family:monospace;">
      &#9209; Stop
    </button>
    <input  id="g_slider" type="range" min="0" max="%%MAX_STEPS_M1%%" value="0"
            style="flex:1;min-width:120px;cursor:pointer;">
    <span   id="g_stepLabel" style="min-width:80px;font-size:13px;">
      Step 1 / %%MAX_STEPS%%
    </span>
    <select id="g_speed" style="padding:4px;background:#2d2d2d;color:white;
                                border:1px solid #555;border-radius:4px;font-family:monospace;">
      <option value="450">Slow</option>
      <option value="150" %%SEL_MEDIUM%%>Medium</option>
      <option value="40"  %%SEL_FAST%%>Fast</option>
      <option value="8"   %%SEL_LIGHTNING%%>Lightning</option>
    </select>
    <span id="g_lossLabel" style="font-size:12px;color:#aaaaaa;margin-left:4px;"></span>
  </div>

  <!-- 3-D surface -->
  <div id="g_plot3d" style="width:100%;height:520px;"></div>

  <!-- 2-D contour + loss curve -->
  <div id="g_plot2d" style="width:100%;height:320px;"></div>

</div>

<!-- Load Plotly synchronously. The next <script> block runs only after
     this one finishes, so Plotly is guaranteed to be defined by then. -->
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(function () {

// ─── Data injected by Python ────────────────────────────────────────────────
var _surfX       = %%SURF_X%%;
var _surfY       = %%SURF_Y%%;
var _surfZ       = %%SURF_Z%%;
var _ctX         = %%CT_X%%;
var _ctY         = %%CT_Y%%;
var _paths       = %%PATHS%%;
var _colors      = %%COLORS%%;
var _convAt      = %%CONV_AT%%;
var _maxSteps    = %%MAX_STEPS%%;
var _xRange      = %%X_RANGE%%;
var _yRange      = %%Y_RANGE%%;
var _surfName    = %%SURF_NAME%%;
var _initDelay   = %%INIT_DELAY%%;

// ─── Animation state ────────────────────────────────────────────────────────
var _step    = 0;
var _playing = false;
var _timer   = null;
var _delay   = _initDelay;

// ─── DOM refs ───────────────────────────────────────────────────────────────
var _playBtn    = document.getElementById('g_play');
var _slider     = document.getElementById('g_slider');
var _stepLabel  = document.getElementById('g_stepLabel');
var _lossLabel  = document.getElementById('g_lossLabel');
var _speedSel   = document.getElementById('g_speed');

// ─── Marker helper ──────────────────────────────────────────────────────────
function _markers(end, color) {
  var sz = [], cl = [], sy = [];
  for (var i = 0; i < end - 1; i++) { sz.push(4);  cl.push(color);   sy.push('circle');  }
  if (end > 0)                       { sz.push(14); cl.push('white'); sy.push('diamond'); }
  return { size: sz, color: cl, symbol: sy };
}

// ─── Trace builders ─────────────────────────────────────────────────────────
function _traces3D(k) {
  var ts = [{
    type: 'surface', x: _surfX, y: _surfY, z: _surfZ,
    colorscale: 'Viridis', opacity: 0.78, showscale: true,
    colorbar: { title: 'Loss', thickness: 12 },
    showlegend: false,
    hovertemplate: 'x:%{x:.2f}<br>y:%{y:.2f}<br>loss:%{z:.4f}<extra></extra>'
  }];
  for (var name in _paths) {
    var p   = _paths[name];
    var end = Math.min(k + 1, p.x.length);
    if (end === 0) continue;
    var m = _markers(end, _colors[name]);
    ts.push({
      type: 'scatter3d',
      x: p.x.slice(0, end), y: p.y.slice(0, end), z: p.z.slice(0, end),
      mode: 'lines+markers',
      line:   { color: _colors[name], width: 5 },
      marker: { size: m.size, color: m.color, symbol: m.symbol },
      name: name, showlegend: true, legendgroup: name
    });
  }
  return ts;
}

function _traces2D(k) {
  var ts = [{
    type: 'contour', x: _ctX, y: _ctY, z: _surfZ,
    colorscale: 'Viridis', showscale: false, ncontours: 25,
    contours: { coloring: 'heatmap' }, opacity: 0.88,
    showlegend: false, xaxis: 'x', yaxis: 'y',
    hovertemplate: 'x:%{x:.2f}<br>y:%{y:.2f}<br>loss:%{z:.4f}<extra></extra>'
  }];

  // Collect all losses for the step-marker y-range
  var allZ = [];
  for (var n in _paths) { allZ = allZ.concat(_paths[n].z); }
  var zMin = Math.min.apply(null, allZ);
  var zMax = Math.max.apply(null, allZ);

  for (var name in _paths) {
    var p   = _paths[name];
    var end = Math.min(k + 1, p.x.length);
    if (end === 0) continue;
    var c = _colors[name];

    // 2-D contour trail
    var sz2 = [], co2 = [], sy2 = [];
    for (var i = 0; i < end - 1; i++) { sz2.push(3);  co2.push(c);       sy2.push('circle'); }
    if (end > 0)                       { sz2.push(10); co2.push('white'); sy2.push('star');   }
    ts.push({
      type: 'scatter', x: p.x.slice(0, end), y: p.y.slice(0, end),
      mode: 'lines+markers',
      line: { color: c, width: 2 }, marker: { size: sz2, color: co2, symbol: sy2 },
      name: name, showlegend: true, legendgroup: name, xaxis: 'x', yaxis: 'y'
    });

    // Loss curve — faded ghost (full path)
    var allSteps = Array.from({ length: p.x.length }, function(_, i) { return i; });
    ts.push({
      type: 'scatter', x: allSteps, y: p.z,
      mode: 'lines', line: { color: c, width: 1.5, dash: 'dot' }, opacity: 0.28,
      showlegend: false, legendgroup: name, hoverinfo: 'skip', xaxis: 'x2', yaxis: 'y2'
    });

    // Loss curve — solid active portion
    var activeSteps = Array.from({ length: end }, function(_, i) { return i; });
    ts.push({
      type: 'scatter', x: activeSteps, y: p.z.slice(0, end),
      mode: 'lines+markers', line: { color: c, width: 2.5 }, marker: { size: 3, color: c },
      showlegend: false, legendgroup: name,
      hovertemplate: name + '<br>step:%{x}<br>loss:%{y:.4f}<extra></extra>',
      xaxis: 'x2', yaxis: 'y2'
    });
  }

  // Vertical step marker
  ts.push({
    type: 'scatter', x: [k, k], y: [zMin, zMax],
    mode: 'lines', line: { color: 'rgba(255,255,255,0.45)', width: 1.5, dash: 'dash' },
    showlegend: false, hoverinfo: 'skip', xaxis: 'x2', yaxis: 'y2'
  });
  return ts;
}

// ─── Plotly layouts ─────────────────────────────────────────────────────────
var _lay3d = {
  // uirevision on BOTH layout and scene is required.
  // layout.uirevision alone does not propagate to the scene in all versions.
  uirevision: 'camera',
  scene: {
    uirevision: 'camera',   // ← locks the camera; never reset on Plotly.react()
    xaxis: { title: 'x', range: _xRange, backgroundcolor: 'rgba(0,0,0,0)', gridcolor: 'rgba(255,255,255,0.12)' },
    yaxis: { title: 'y', range: _yRange, backgroundcolor: 'rgba(0,0,0,0)', gridcolor: 'rgba(255,255,255,0.12)' },
    zaxis: { title: 'Loss',              backgroundcolor: 'rgba(0,0,0,0)', gridcolor: 'rgba(255,255,255,0.12)' },
    bgcolor: 'rgba(17,17,17,1)'
  },
  title:          { text: _surfName + ' — step 0', font: { size: 16, color: 'white' }, x: 0.5 },
  paper_bgcolor:  'rgba(17,17,17,1)',
  font:           { color: 'white', family: 'monospace' },
  legend:         { x: 0.01, y: 0.97, bgcolor: 'rgba(40,40,40,0.85)', bordercolor: 'rgba(255,255,255,0.2)', borderwidth: 1 },
  height:         520,
  margin:         { l: 0, r: 10, t: 45, b: 0 }
};

var _g = 'rgba(255,255,255,0.10)';
var _z = 'rgba(255,255,255,0.20)';
var _lay2d = {
  grid: { rows: 1, columns: 2, pattern: 'independent' },
  xaxis:  { title: 'x',    domain: [0, 0.44], gridcolor: _g, zerolinecolor: _z },
  yaxis:  { title: 'y',                        gridcolor: _g, zerolinecolor: _z },
  xaxis2: { title: 'Step', domain: [0.56, 1],  gridcolor: _g, zerolinecolor: _z },
  yaxis2: { title: 'Loss',                     gridcolor: _g, zerolinecolor: _z },
  paper_bgcolor: 'rgba(17,17,17,1)', plot_bgcolor: 'rgba(17,17,17,1)',
  font:   { color: 'white', family: 'monospace' },
  legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(40,40,40,0.85)', bordercolor: 'rgba(255,255,255,0.2)', borderwidth: 1 },
  annotations: [
    { text: 'Contour View', x: 0.22, xref: 'paper', y: 1.06, yref: 'paper', showarrow: false, font: { size: 12, color: 'rgba(210,210,210,0.9)' } },
    { text: 'Loss vs. Step', x: 0.78, xref: 'paper', y: 1.06, yref: 'paper', showarrow: false, font: { size: 12, color: 'rgba(210,210,210,0.9)' } }
  ],
  height: 320, margin: { l: 10, r: 10, t: 35, b: 10 }
};

// ─── Render a specific step ──────────────────────────────────────────────────
function _render(k) {
  _step = k;
  _slider.value = k;
  _stepLabel.textContent = 'Step ' + (k + 1) + ' / ' + _maxSteps;
  _lay3d.title.text = _surfName + ' \u2014 step ' + k;

  // Live loss labels
  var parts = [];
  for (var name in _paths) {
    var p   = _paths[name];
    var idx = Math.min(k, p.z.length - 1);
    var conv = (_convAt[name] !== undefined && _convAt[name] <= k) ? ' \u2713' : '';
    parts.push(name + ': ' + p.z[idx].toFixed(4) + conv);
  }
  _lossLabel.textContent = parts.join('  |  ');

  // Plotly.react() preserves camera because uirevision hasn't changed
  Plotly.react('g_plot3d', _traces3D(k), _lay3d);
  Plotly.react('g_plot2d', _traces2D(k), _lay2d);
}

// ─── Playback controls ───────────────────────────────────────────────────────
function _tick() {
  if (_step >= _maxSteps - 1) { _pause(); return; }
  _render(_step + 1);
}

function _play() {
  _playing = true;
  _playBtn.textContent = '\u23F8 Pause';
  _timer = setInterval(_tick, _delay);
}

function _pause() {
  _playing = false;
  _playBtn.textContent = '\u25B6 Play';
  clearInterval(_timer);
  _timer = null;
}

function _togglePlay()  { if (_playing) _pause(); else _play(); }
function _stop()        { _pause(); _render(0); }
function _scrub(val)    { if (_playing) _pause(); _render(parseInt(val, 10)); }
function _setSpeed()    {
  _delay = parseInt(_speedSel.value, 10);
  if (_playing) { clearInterval(_timer); _timer = setInterval(_tick, _delay); }
}

// ─── Wire up controls ────────────────────────────────────────────────────────
_playBtn.addEventListener('click', _togglePlay);
document.getElementById('g_stop').addEventListener('click', _stop);
_slider.addEventListener('input', function() { _scrub(this.value); });
_speedSel.addEventListener('change', _setSpeed);

// ─── Initial render ──────────────────────────────────────────────────────────
// newPlot for first draw; all subsequent frames use react() which preserves camera
Plotly.newPlot('g_plot3d', _traces3D(0), _lay3d, { responsive: true });
Plotly.newPlot('g_plot2d', _traces2D(0), _lay2d, { responsive: true });

}()); // end IIFE — Plotly is already loaded by the time this runs
</script>
</body>
</html>
"""


def build_animation_html(
    surface: Surface,
    x_lin, y_lin, X, Y, Z,
    paths: dict,
    converged_at: dict,
    speed: str = "Medium",
) -> str:
    """
    Serialise all path data to JSON and inject it into the HTML/JS template.
    Returns a fully self-contained HTML string ready for gr.HTML.
    """
    # Serialise path data — convert numpy arrays to plain Python lists
    paths_js = {
        name: {
            "x": px.tolist(),
            "y": py.tolist(),
            "z": pz.tolist(),
        }
        for name, (px, py, pz) in paths.items()
    }

    max_steps = max(len(v["x"]) for v in paths_js.values())

    # Only expose colours for optimizers that are actually in use
    colors_js = {name: OPTIMIZER_COLORS[name] for name in paths_js}

    init_delay = SPEED_MS.get(speed, 150)

    # Build the speed dropdown selected-attribute strings
    sel = {s: "" for s in SPEED_MS}
    sel[speed] = "selected"

    inner = (
        _TEMPLATE
        .replace("%%SURF_X%%",        json.dumps(X.tolist()))
        .replace("%%SURF_Y%%",        json.dumps(Y.tolist()))
        .replace("%%SURF_Z%%",        json.dumps(Z.tolist()))
        .replace("%%CT_X%%",          json.dumps(x_lin.tolist()))
        .replace("%%CT_Y%%",          json.dumps(y_lin.tolist()))
        .replace("%%PATHS%%",         json.dumps(paths_js))
        .replace("%%COLORS%%",        json.dumps(colors_js))
        .replace("%%CONV_AT%%",       json.dumps(converged_at))
        .replace("%%MAX_STEPS_M1%%",  str(max_steps - 1))
        .replace("%%MAX_STEPS%%",     str(max_steps))
        .replace("%%X_RANGE%%",       json.dumps(list(surface.x_range)))
        .replace("%%Y_RANGE%%",       json.dumps(list(surface.y_range)))
        .replace("%%SURF_NAME%%",     json.dumps(surface.name))
        .replace("%%INIT_DELAY%%",    str(init_delay))
        .replace("%%SEL_MEDIUM%%",    sel["Medium"])
        .replace("%%SEL_FAST%%",      sel["Fast"])
        .replace("%%SEL_LIGHTNING%%", sel["Lightning"])
    )

    # gr.HTML injects content via innerHTML which does NOT execute <script> tags —
    # this is a browser security restriction that can't be worked around inline.
    # Wrapping in <iframe srcdoc="..."> gives the content its own document context
    # where scripts execute normally, including loading Plotly from CDN.
    # Escaping order: & first (so the & in &quot; isn't double-escaped), then ".
    srcdoc = inner.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:900px;border:none;background:#111111;border-radius:8px;" '
        f'></iframe>'
    )
