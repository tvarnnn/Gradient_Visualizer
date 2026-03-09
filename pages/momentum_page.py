"""
Momentum Dynamics — 1D loss surface animation comparing SGD, Momentum, and Adam.
Focuses on showing WHY momentum helps: smoother paths, faster convergence,
better escape from ravines.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import gradio as gr
from optimizers import SGD, Momentum as MomentumOpt, Adam
from pages.shared import THEME, BACK_HTML

# ── 1-D surfaces ─────────────────────────────────────────────────────────────
SURFACES_1D = {
    "Bumpy Bowl": {
        "f":    lambda x: x**2 + 2.0 * np.sin(3 * x),
        "grad": lambda x: 2*x + 6.0 * np.cos(3 * x),
        "xrange": (-4.5, 4.5),
        "x0_default": -3.8,
    },
    "Steep Ravine": {
        "f":    lambda x: 50 * (x - 1)**2,
        "grad": lambda x: 100 * (x - 1),
        "xrange": (-0.5, 2.8),
        "x0_default": -0.3,
    },
    "Quadratic": {
        "f":    lambda x: x**2,
        "grad": lambda x: 2*x,
        "xrange": (-4.5, 4.5),
        "x0_default": -4.0,
    },
    "Multi-modal": {
        "f":    lambda x: np.sin(x) + 0.05 * x**2,
        "grad": lambda x: np.cos(x) + 0.10 * x,
        "xrange": (-10, 10),
        "x0_default": 8.5,
    },
}

SURFACE_NAMES = list(SURFACES_1D.keys())
OPT_COLORS = {"SGD": "#e74c3c", "Momentum": "#e67e22", "Adam": "#9b59b6"}

CONVERGENCE_TOL = 1e-7
X_CURVE_N = 600

_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>* { box-sizing:border-box;margin:0;padding:0; } body { background:#111111; }</style>
</head><body>
<div style="background:#111111;color:white;font-family:monospace;border-radius:8px;overflow:hidden;">

  <!-- Controls -->
  <div style="padding:8px 12px;display:flex;align-items:center;gap:10px;
              background:#1a1a2e;flex-wrap:wrap;">
    <button id="m_play" style="padding:5px 14px;background:#7c3aed;color:white;
                               border:none;border-radius:4px;cursor:pointer;font-family:monospace;">
      &#9654; Play
    </button>
    <button id="m_stop" style="padding:5px 14px;background:#dc2626;color:white;
                               border:none;border-radius:4px;cursor:pointer;font-family:monospace;">
      &#9209; Stop
    </button>
    <input id="m_slider" type="range" min="0" max="%%MAX_M1%%" value="0"
           style="flex:1;min-width:120px;cursor:pointer;">
    <span id="m_label" style="min-width:90px;font-size:13px;">Step 1 / %%MAX%%</span>
    <select id="m_speed" style="padding:4px;background:#2d2d2d;color:white;
                                border:1px solid #555;border-radius:4px;font-family:monospace;">
      <option value="450">Slow</option>
      <option value="150" %%SEL_MEDIUM%%>Medium</option>
      <option value="40"  %%SEL_FAST%%>Fast</option>
      <option value="8"   %%SEL_LIGHTNING%%>Lightning</option>
    </select>
    <span id="m_loss_label" style="font-size:12px;color:#aaa;margin-left:4px;"></span>
  </div>

  <!-- Surface -->
  <div id="m_surf" style="width:100%;height:390px;"></div>
  <!-- Loss curve -->
  <div id="m_loss" style="width:100%;height:270px;"></div>

</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(function () {
var _sx    = %%SURF_X%%;
var _sy    = %%SURF_Y%%;
var _paths = %%PATHS%%;
var _cols  = %%COLORS%%;
var _names = %%NAMES%%;
var _max   = %%MAX%%;
var _sname = %%SURF_NAME%%;
var _delay = %%DELAY%%;

var _step=0, _playing=false, _timer=null;

var _playBtn = document.getElementById('m_play');
var _slider  = document.getElementById('m_slider');
var _label   = document.getElementById('m_label');
var _ll      = document.getElementById('m_loss_label');
var _speedSel= document.getElementById('m_speed');

var _G = 'rgba(255,255,255,0.09)';
var _Z = 'rgba(255,255,255,0.18)';

var _laySurf = {
  uirevision: 'surf',
  title: { text: _sname+' \u2014 step 0', font:{size:15,color:'white'}, x:0.5 },
  paper_bgcolor:'rgba(17,17,17,1)', plot_bgcolor:'rgba(17,17,17,1)',
  font:{color:'white',family:'monospace'},
  xaxis:{ title:'x', gridcolor:_G, zerolinecolor:_Z },
  yaxis:{ title:'Loss', gridcolor:_G, zerolinecolor:_Z },
  legend:{ x:0.01,y:0.99,bgcolor:'rgba(40,40,40,0.85)',
           bordercolor:'rgba(255,255,255,0.2)',borderwidth:1 },
  height:390, margin:{l:55,r:20,t:45,b:40}
};

var _layLoss = {
  uirevision: 'loss',
  title:{ text:'Loss vs Step', font:{size:13,color:'white'}, x:0.5 },
  paper_bgcolor:'rgba(17,17,17,1)', plot_bgcolor:'rgba(17,17,17,1)',
  font:{color:'white',family:'monospace'},
  xaxis:{ title:'Step', gridcolor:_G, zerolinecolor:_Z },
  yaxis:{ title:'Loss', gridcolor:_G, zerolinecolor:_Z },
  height:270, margin:{l:55,r:20,t:35,b:40}
};

function surfTraces(k) {
  var ts = [{
    type:'scatter', x:_sx, y:_sy, mode:'lines',
    line:{color:'#a78bfa',width:2.5}, name:_sname, showlegend:true
  }];
  for (var i=0; i<_names.length; i++) {
    var nm = _names[i], p = _paths[nm], c = _cols[nm];
    var end = Math.min(k+1, p.x.length);
    if (end === 0) continue;
    if (end > 1) {
      ts.push({
        type:'scatter', x:p.x.slice(0,end-1), y:p.y.slice(0,end-1),
        mode:'markers', marker:{size:4,color:c,opacity:0.45},
        showlegend:false, legendgroup:nm, hoverinfo:'skip'
      });
    }
    ts.push({
      type:'scatter', x:[p.x[end-1]], y:[p.y[end-1]], mode:'markers',
      marker:{size:15,color:c,symbol:'circle',line:{color:'white',width:1.5}},
      name:nm, legendgroup:nm, showlegend:true,
      hovertemplate: nm+'<br>x: %{x:.4f}<br>loss: %{y:.4f}<extra></extra>'
    });
  }
  return ts;
}

function lossTraces(k) {
  var ts = [];
  var allY = [];
  for (var i=0; i<_names.length; i++) allY = allY.concat(_paths[_names[i]].y);
  var yMin = Math.min.apply(null,allY), yMax = Math.max.apply(null,allY);

  for (var i=0; i<_names.length; i++) {
    var nm = _names[i], p = _paths[nm], c = _cols[nm];
    var end = Math.min(k+1, p.x.length);
    var steps = p.x.map(function(_,j){return j;});
    ts.push({
      type:'scatter', x:steps, y:p.y, mode:'lines',
      line:{color:c,width:1,dash:'dot'}, opacity:0.22,
      showlegend:false, legendgroup:nm, hoverinfo:'skip'
    });
    ts.push({
      type:'scatter', x:steps.slice(0,end), y:p.y.slice(0,end),
      mode:'lines+markers', line:{color:c,width:2.5}, marker:{size:3,color:c},
      name:nm, legendgroup:nm, showlegend:false,
      hovertemplate:nm+'<br>step:%{x}<br>loss:%{y:.4f}<extra></extra>'
    });
  }
  ts.push({
    type:'scatter', x:[k,k], y:[yMin,yMax], mode:'lines',
    line:{color:'rgba(255,255,255,0.4)',width:1.5,dash:'dash'},
    showlegend:false, hoverinfo:'skip'
  });
  return ts;
}

function render(k) {
  _step = k;
  _slider.value = k;
  _label.textContent = 'Step '+(k+1)+' / '+_max;
  _laySurf.title.text = _sname+' \u2014 step '+k;
  var parts = [];
  for (var i=0; i<_names.length; i++) {
    var nm=_names[i], p=_paths[nm];
    var idx=Math.min(k,p.y.length-1);
    parts.push(nm+': '+p.y[idx].toFixed(5));
  }
  _ll.textContent = parts.join('  |  ');
  Plotly.react('m_surf', surfTraces(k), _laySurf);
  Plotly.react('m_loss', lossTraces(k), _layLoss);
}

function tick()  { if (_step>=_max-1){pause();return;} render(_step+1); }
function play()  { _playing=true;  _playBtn.textContent='\u23F8 Pause'; _timer=setInterval(tick,_delay); }
function pause() { _playing=false; _playBtn.textContent='\u25B6 Play';  clearInterval(_timer);_timer=null; }
function togglePlay() { if(_playing) pause(); else play(); }
function stop()  { pause(); render(0); }

_playBtn.addEventListener('click', togglePlay);
document.getElementById('m_stop').addEventListener('click', stop);
_slider.addEventListener('input', function(){if(_playing)pause();render(parseInt(this.value,10));});
_speedSel.addEventListener('change', function(){
  _delay=parseInt(this.value,10);
  if(_playing){clearInterval(_timer);_timer=setInterval(tick,_delay);}
});

Plotly.newPlot('m_surf', surfTraces(0), _laySurf, {responsive:true});
Plotly.newPlot('m_loss', lossTraces(0), _layLoss, {responsive:true});
}());
</script>
</body></html>"""


def _make_srcdoc(html: str) -> str:
    srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:730px;border:none;background:#111111;border-radius:8px;">'
        f'</iframe>'
    )


def run_momentum(surface_name, lr, beta, x0, n_steps, speed):
    surf = SURFACES_1D[surface_name]
    f, grad = surf["f"], surf["grad"]
    xr = surf["xrange"]

    # Surface curve
    x_curve = np.linspace(xr[0], xr[1], X_CURVE_N)
    y_curve  = np.array([f(x) for x in x_curve])

    # Run optimizers
    opts = {
        "SGD":      SGD(lr),
        "Momentum": MomentumOpt(lr, beta=beta),
        "Adam":     Adam(lr),
    }
    paths = {}
    for nm, opt in opts.items():
        opt.reset()
        params = np.array([float(x0)])
        xs, ys = [params[0]], [float(f(params[0]))]
        for _ in range(int(n_steps)):
            g = np.array([grad(params[0])])
            if abs(g[0]) < CONVERGENCE_TOL or abs(g[0]) > 1e7:
                break
            params = opt.step(params, g)
            params = np.clip(params, xr[0], xr[1])
            xs.append(float(params[0]))
            ys.append(float(f(params[0])))
        paths[nm] = {"x": xs, "y": ys}

    max_steps = max(len(v["x"]) for v in paths.values())

    SPEED_MS = {"Slow": 450, "Medium": 150, "Fast": 40, "Lightning": 8}
    sel = {s: "" for s in SPEED_MS}
    sel[speed] = "selected"

    html = (
        _TEMPLATE
        .replace("%%SURF_X%%",    json.dumps(x_curve.tolist()))
        .replace("%%SURF_Y%%",    json.dumps(y_curve.tolist()))
        .replace("%%PATHS%%",     json.dumps(paths))
        .replace("%%COLORS%%",    json.dumps(OPT_COLORS))
        .replace("%%NAMES%%",     json.dumps(list(opts.keys())))
        .replace("%%MAX_M1%%",    str(max_steps - 1))
        .replace("%%MAX%%",       str(max_steps))
        .replace("%%SURF_NAME%%", json.dumps(surface_name))
        .replace("%%DELAY%%",     str(SPEED_MS.get(speed, 150)))
        .replace("%%SEL_MEDIUM%%",    sel["Medium"])
        .replace("%%SEL_FAST%%",      sel["Fast"])
        .replace("%%SEL_LIGHTNING%%", sel["Lightning"])
    )
    return _make_srcdoc(html)


def on_surface_change(name):
    return SURFACES_1D[name]["x0_default"]


with gr.Blocks(title="Momentum Dynamics — ML Playground") as demo:
    gr.HTML(BACK_HTML)
    gr.Markdown(
        "# Momentum Dynamics\n"
        "### SGD vs Momentum vs Adam on a 1D loss surface — see exactly how velocity changes the path"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            surf_dd  = gr.Dropdown(choices=SURFACE_NAMES, value="Bumpy Bowl", label="Loss Surface")
            x0_sl    = gr.Slider(-10, 10, step=0.1, value=-3.8, label="Starting x₀")
            lr_sl    = gr.Slider(0.001, 0.3, step=0.001, value=0.02, label="Learning Rate")
            beta_sl  = gr.Slider(0.0, 0.99, step=0.01, value=0.9,
                                 label="Momentum β",
                                 info="Used by the Momentum optimizer only")
            steps_sl = gr.Slider(10, 300, step=10, value=80, label="Steps")
            speed_dd = gr.Dropdown(choices=["Slow","Medium","Fast","Lightning"],
                                   value="Medium", label="Playback Speed")
            run_btn  = gr.Button("▶ Run", variant="primary", size="lg")

        with gr.Column(scale=2):
            anim_out = gr.HTML(
                "<p style='color:#888;font-family:monospace;padding:20px'>"
                "Configure and click <b>▶ Run</b> to start.</p>"
            )

    surf_dd.change(fn=on_surface_change, inputs=[surf_dd], outputs=[x0_sl])
    run_btn.click(
        fn=run_momentum,
        inputs=[surf_dd, lr_sl, beta_sl, x0_sl, steps_sl, speed_dd],
        outputs=[anim_out],
    )
