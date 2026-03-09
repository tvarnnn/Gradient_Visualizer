"""
Neural Network Trainer — train a numpy MLP and watch the decision boundary evolve.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import gradio as gr
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from pages.shared import THEME, BACK_HTML

DATASETS = ["Moons", "Circles", "XOR", "Spirals"]
MESH_N   = 80          # mesh resolution for decision boundary
RECORD_EVERY = 5       # record frame every N epochs

# ── Tiny numpy MLP ────────────────────────────────────────────────────────────

class MLP:
    def __init__(self, hidden=16, activation="relu", seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(2, hidden) * np.sqrt(2.0 / 2)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, 1) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1)
        self.act = activation

    def _act(self, z):
        if self.act == "relu":
            return np.maximum(0, z)
        elif self.act == "tanh":
            return np.tanh(z)
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # sigmoid

    def _act_grad(self, z):
        if self.act == "relu":
            return (z > 0).astype(float)
        elif self.act == "tanh":
            t = np.tanh(z)
            return 1 - t * t
        s = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        return s * (1 - s)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._act(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = 1 / (1 + np.exp(-np.clip(self.z2, -250, 250)))
        return self.a2.ravel()

    def loss(self, X, y):
        p = self.forward(X)
        eps = 1e-8
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def step(self, X, y, lr):
        m  = X.shape[0]
        p  = self.forward(X)
        dz2 = (p - y).reshape(-1, 1) / m
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._act_grad(self.z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0)
        self.W1 -= lr * dW1;  self.b1 -= lr * db1
        self.W2 -= lr * dW2;  self.b2 -= lr * db2


# ── Dataset generators ────────────────────────────────────────────────────────

def make_dataset(name, n, noise, seed=42):
    rng = np.random.RandomState(seed)
    if name == "Moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    elif name == "Circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.45, random_state=seed)
    elif name == "XOR":
        X = rng.randn(n, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X += rng.randn(n, 2) * noise
    else:   # Spirals
        def spiral_branch(n, delta, rng):
            t = np.linspace(0, 4 * np.pi, n)
            r = t / (4 * np.pi)
            x = r * np.cos(t + delta) + rng.randn(n) * noise * 0.5
            y = r * np.sin(t + delta) + rng.randn(n) * noise * 0.5
            return np.c_[x, y]
        X = np.vstack([spiral_branch(n // 2, 0, rng), spiral_branch(n // 2, np.pi, rng)])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y.astype(float)


# ── HTML animation template ───────────────────────────────────────────────────

_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>* { box-sizing:border-box;margin:0;padding:0; } body { background:#111111; }</style>
</head><body>
<div style="background:#111111;color:white;font-family:monospace;border-radius:8px;overflow:hidden;">

  <div style="padding:8px 12px;display:flex;align-items:center;gap:10px;
              background:#1a1a2e;flex-wrap:wrap;">
    <button id="n_play" style="padding:5px 14px;background:#7c3aed;color:white;
                               border:none;border-radius:4px;cursor:pointer;font-family:monospace;">
      &#9654; Play
    </button>
    <button id="n_stop" style="padding:5px 14px;background:#dc2626;color:white;
                               border:none;border-radius:4px;cursor:pointer;font-family:monospace;">
      &#9209; Stop
    </button>
    <input id="n_slider" type="range" min="0" max="%%MAX_M1%%" value="0"
           style="flex:1;min-width:120px;cursor:pointer;">
    <span id="n_label" style="min-width:100px;font-size:13px;">Epoch 0 / %%TOTAL_EPOCHS%%</span>
    <select id="n_speed" style="padding:4px;background:#2d2d2d;color:white;
                                border:1px solid #555;border-radius:4px;font-family:monospace;">
      <option value="600">Slow</option>
      <option value="200" %%SEL_MED%%>Medium</option>
      <option value="60" %%SEL_FAST%%>Fast</option>
      <option value="12">Lightning</option>
    </select>
    <span id="n_loss_label" style="font-size:12px;color:#aaa;"></span>
  </div>

  <div style="display:flex;">
    <div id="n_boundary" style="flex:1;height:460px;"></div>
    <div id="n_losscurve" style="width:320px;height:460px;"></div>
  </div>

</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(function () {
var _meshX   = %%MESH_X%%;
var _meshY   = %%MESH_Y%%;
var _frames  = %%FRAMES%%;
var _dataX0  = %%DATA_X0%%;
var _dataX1  = %%DATA_X1%%;
var _labels  = %%LABELS%%;
var _epochs  = %%EPOCHS%%;
var _totalEp = %%TOTAL_EPOCHS%%;
var _max     = %%MAX%%;
var _delay   = %%DELAY%%;

var _step=0, _playing=false, _timer=null;
var _playBtn = document.getElementById('n_play');
var _slider  = document.getElementById('n_slider');
var _label   = document.getElementById('n_label');
var _ll      = document.getElementById('n_loss_label');
var _speedSel= document.getElementById('n_speed');

var _G = 'rgba(255,255,255,0.09)';

// Static scatter traces for data
var _scatter0 = {
  type:'scatter', x:[], y:[], mode:'markers',
  marker:{color:'#f87171',size:7,line:{color:'white',width:0.6},opacity:0.9},
  name:'Class 0', showlegend:true
};
var _scatter1 = {
  type:'scatter', x:[], y:[], mode:'markers',
  marker:{color:'#60a5fa',size:7,line:{color:'white',width:0.6},opacity:0.9},
  name:'Class 1', showlegend:true
};
for (var i=0; i<_labels.length; i++) {
  if (_labels[i]===0) { _scatter0.x.push(_dataX0[i]); _scatter0.y.push(_dataX1[i]); }
  else                { _scatter1.x.push(_dataX0[i]); _scatter1.y.push(_dataX1[i]); }
}

var _layB = {
  uirevision:'boundary',
  title:{text:'Decision Boundary — epoch 0', font:{size:14,color:'white'}, x:0.5},
  paper_bgcolor:'rgba(17,17,17,1)', plot_bgcolor:'rgba(17,17,17,1)',
  font:{color:'white',family:'monospace'},
  xaxis:{title:'x₁',gridcolor:_G,zeroline:false},
  yaxis:{title:'x₂',gridcolor:_G,zeroline:false,scaleanchor:'x'},
  legend:{x:0.01,y:0.99,bgcolor:'rgba(40,40,40,0.85)',bordercolor:'rgba(255,255,255,0.2)',borderwidth:1},
  height:460, margin:{l:50,r:10,t:45,b:40}
};

var _layL = {
  uirevision:'loss',
  title:{text:'Training Loss', font:{size:13,color:'white'}, x:0.5},
  paper_bgcolor:'rgba(17,17,17,1)', plot_bgcolor:'rgba(17,17,17,1)',
  font:{color:'white',family:'monospace'},
  xaxis:{title:'Epoch',gridcolor:_G,zeroline:false},
  yaxis:{title:'Loss',gridcolor:_G,zeroline:false},
  showlegend:false,
  height:460, margin:{l:55,r:15,t:45,b:40}
};

function allLosses() {
  return _frames.map(function(f){return f.loss;});
}

function boundaryTraces(k) {
  return [{
    type:'heatmap', x:_meshX, y:_meshY, z:_frames[k].probs,
    colorscale:[[0,'rgba(231,76,60,0.55)'],[0.5,'rgba(150,150,150,0.05)'],[1,'rgba(52,152,219,0.55)']],
    zmin:0, zmax:1, showscale:true,
    colorbar:{title:'P(class 1)',thickness:10,len:0.6},
    hoverinfo:'skip'
  }, _scatter0, _scatter1];
}

function lossTraces(k) {
  var allL = allLosses();
  var eps  = _epochs.slice(0, k+1);
  var ls   = allL.slice(0, k+1);
  return [
    {type:'scatter', x:_epochs, y:allL, mode:'lines',
     line:{color:'rgba(167,139,250,0.2)',width:1,dash:'dot'},hoverinfo:'skip'},
    {type:'scatter', x:eps, y:ls, mode:'lines+markers',
     line:{color:'#a78bfa',width:2.5}, marker:{size:3,color:'#a78bfa'},
     hovertemplate:'epoch %{x}<br>loss %{y:.4f}<extra></extra>'},
    {type:'scatter', x:[_epochs[k]], y:[allL[k]], mode:'markers',
     marker:{size:9,color:'white',symbol:'circle'}},
  ];
}

function render(k) {
  _step=k; _slider.value=k;
  _label.textContent = 'Epoch '+_epochs[k]+' / '+_totalEp;
  _ll.textContent    = 'loss: '+_frames[k].loss.toFixed(5);
  _layB.title.text   = 'Decision Boundary \u2014 epoch '+_epochs[k];
  Plotly.react('n_boundary',  boundaryTraces(k), _layB);
  Plotly.react('n_losscurve', lossTraces(k),     _layL);
}

function tick()  { if(_step>=_max-1){pause();return;} render(_step+1); }
function play()  { _playing=true;  _playBtn.textContent='\u23F8 Pause'; _timer=setInterval(tick,_delay); }
function pause() { _playing=false; _playBtn.textContent='\u25B6 Play';  clearInterval(_timer);_timer=null; }
function stop()  { pause(); render(0); }
function togglePlay() { if(_playing) pause(); else play(); }

_playBtn.addEventListener('click', togglePlay);
document.getElementById('n_stop').addEventListener('click', stop);
_slider.addEventListener('input', function(){if(_playing)pause();render(parseInt(this.value,10));});
_speedSel.addEventListener('change', function(){
  _delay=parseInt(this.value,10);
  if(_playing){clearInterval(_timer);_timer=setInterval(tick,_delay);}
});

Plotly.newPlot('n_boundary',  boundaryTraces(0), _layB, {responsive:true});
Plotly.newPlot('n_losscurve', lossTraces(0),     _layL, {responsive:true});
}());
</script>
</body></html>"""


def _srcdoc(html: str) -> str:
    srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:560px;border:none;background:#111111;border-radius:8px;">'
        f'</iframe>'
    )


def run_nn(dataset, n_points, noise, hidden, activation, lr, total_epochs, speed):
    X, y = make_dataset(dataset, int(n_points), noise)

    # Mesh for decision boundary
    pad = 0.4
    x1r = (X[:, 0].min() - pad, X[:, 0].max() + pad)
    x2r = (X[:, 1].min() - pad, X[:, 1].max() + pad)
    mx  = np.linspace(*x1r, MESH_N)
    my  = np.linspace(*x2r, MESH_N)
    gxx, gyy = np.meshgrid(mx, my)
    grid = np.c_[gxx.ravel(), gyy.ravel()]

    net = MLP(hidden=int(hidden), activation=activation)
    frames = []
    epoch_labels = []

    SPEED_MS = {"Slow": 600, "Medium": 200, "Fast": 60, "Lightning": 12}
    sel = {s: "" for s in SPEED_MS}
    sel[speed] = "selected"

    for ep in range(int(total_epochs) + 1):
        if ep % RECORD_EVERY == 0 or ep == int(total_epochs):
            probs = net.forward(grid).reshape(MESH_N, MESH_N)
            loss  = net.loss(X, y)
            frames.append({"probs": probs.tolist(), "loss": loss})
            epoch_labels.append(ep)
        if ep < int(total_epochs):
            net.step(X, y, lr)

    html = (
        _TEMPLATE
        .replace("%%MESH_X%%",      json.dumps(mx.tolist()))
        .replace("%%MESH_Y%%",      json.dumps(my.tolist()))
        .replace("%%FRAMES%%",      json.dumps(frames))
        .replace("%%DATA_X0%%",     json.dumps(X[:, 0].tolist()))
        .replace("%%DATA_X1%%",     json.dumps(X[:, 1].tolist()))
        .replace("%%LABELS%%",      json.dumps(y.astype(int).tolist()))
        .replace("%%EPOCHS%%",      json.dumps(epoch_labels))
        .replace("%%TOTAL_EPOCHS%%", str(int(total_epochs)))
        .replace("%%MAX_M1%%",      str(len(frames) - 1))
        .replace("%%MAX%%",         str(len(frames)))
        .replace("%%DELAY%%",       str(SPEED_MS.get(speed, 200)))
        .replace("%%SEL_MED%%",     sel["Medium"])
        .replace("%%SEL_FAST%%",    sel["Fast"])
    )
    return _srcdoc(html)


with gr.Blocks(title="NN Trainer — ML Playground") as demo:
    gr.HTML(BACK_HTML)
    gr.Markdown(
        "# Neural Network Trainer\n"
        "### Train a numpy MLP and watch the decision boundary evolve in real time"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            dataset_dd  = gr.Dropdown(choices=DATASETS, value="Moons", label="Dataset")
            n_pts_sl    = gr.Slider(100, 500, step=50, value=200, label="Points")
            noise_sl    = gr.Slider(0.0, 0.4, step=0.02, value=0.15, label="Noise")
            hidden_sl   = gr.Slider(4, 64, step=4, value=16, label="Hidden Units")
            act_dd      = gr.Dropdown(choices=["relu", "tanh", "sigmoid"],
                                      value="relu", label="Activation")
            lr_sl       = gr.Slider(0.001, 0.5, step=0.001, value=0.05, label="Learning Rate")
            epochs_sl   = gr.Slider(50, 1000, step=50, value=200, label="Epochs")
            speed_dd    = gr.Dropdown(choices=["Slow","Medium","Fast","Lightning"],
                                      value="Medium", label="Playback Speed")
            run_btn     = gr.Button("▶ Train", variant="primary", size="lg")

        with gr.Column(scale=2):
            anim_out = gr.HTML(
                "<p style='color:#888;font-family:monospace;padding:20px'>"
                "Configure and click <b>▶ Train</b> to start.</p>"
            )

    run_btn.click(
        fn=run_nn,
        inputs=[dataset_dd, n_pts_sl, noise_sl, hidden_sl, act_dd, lr_sl, epochs_sl, speed_dd],
        outputs=[anim_out],
    )
