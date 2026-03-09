"""
RL Visualizer — value iteration on a grid world.
Shows the value function heatmap and greedy policy arrows evolving each sweep.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import gradio as gr
from pages.shared import THEME, BACK_HTML

# ── Value Iteration ───────────────────────────────────────────────────────────
# Actions: 0=up(row-1), 1=down(row+1), 2=left(col-1), 3=right(col+1)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def value_iteration(n, goal, obstacles, gamma, n_iter, step_cost):
    """
    Returns (v_frames, p_frames, delta_history).
    v_frames : list[ndarray(n,n)]  — V at each iteration (display-flipped)
    p_frames : list[list[int]]     — policy flat list, -1 = terminal
    deltas   : list[float]         — max |ΔV| per iteration
    """
    goal      = tuple(goal)
    obstacles = set(tuple(o) for o in obstacles)
    terminal  = obstacles | {goal}

    R = np.full((n, n), step_cost)
    R[goal]    = 1.0
    for obs in obstacles:
        R[obs] = -1.0

    V = np.zeros((n, n))

    v_frames, p_frames, deltas = [], [], []

    def greedy_policy():
        P = np.full(n * n, -1, dtype=int)
        for i in range(n):
            for j in range(n):
                if (i, j) in terminal:
                    continue
                best_q, best_a = -1e9, 0
                for ai, (di, dj) in enumerate(ACTIONS):
                    ni2 = max(0, min(n - 1, i + di))
                    nj2 = max(0, min(n - 1, j + dj))
                    q   = R[i, j] + gamma * V[ni2, nj2]
                    if q > best_q:
                        best_q, best_a = q, ai
                P[i * n + j] = best_a
        return P.tolist()

    # Frame 0 (initial)
    v_frames.append(np.flipud(V).tolist())
    p_frames.append(greedy_policy())
    deltas.append(0.0)

    for _ in range(n_iter):
        V_new = V.copy()
        for i in range(n):
            for j in range(n):
                if (i, j) in terminal:
                    V_new[i, j] = R[i, j]
                    continue
                best_q = -1e9
                for di, dj in ACTIONS:
                    ni2 = max(0, min(n - 1, i + di))
                    nj2 = max(0, min(n - 1, j + dj))
                    best_q = max(best_q, R[i, j] + gamma * V[ni2, nj2])
                V_new[i, j] = best_q
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        v_frames.append(np.flipud(V).tolist())
        p_frames.append(greedy_policy())
        deltas.append(delta)
        if delta < 1e-6:
            break

    return v_frames, p_frames, deltas


# ── HTML animation template ───────────────────────────────────────────────────

_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>* { box-sizing:border-box;margin:0;padding:0; } body { background:#111111; }</style>
</head><body>
<div style="background:#111111;color:white;font-family:monospace;border-radius:8px;overflow:hidden;">

  <div style="padding:8px 12px;display:flex;align-items:center;gap:10px;
              background:#1a1a2e;flex-wrap:wrap;">
    <button id="r_play" style="padding:5px 14px;background:#7c3aed;color:white;
                               border:none;border-radius:4px;cursor:pointer;font-family:monospace;">
      &#9654; Play
    </button>
    <button id="r_stop" style="padding:5px 14px;background:#dc2626;color:white;
                               border:none;border-radius:4px;cursor:pointer;font-family:monospace;">
      &#9209; Stop
    </button>
    <input id="r_slider" type="range" min="0" max="%%MAX_M1%%" value="0"
           style="flex:1;min-width:120px;cursor:pointer;">
    <span id="r_label" style="min-width:100px;font-size:13px;">Sweep 0 / %%MAX%%</span>
    <select id="r_speed" style="padding:4px;background:#2d2d2d;color:white;
                                border:1px solid #555;border-radius:4px;font-family:monospace;">
      <option value="800">Slow</option>
      <option value="300" %%SEL_MED%%>Medium</option>
      <option value="80" %%SEL_FAST%%>Fast</option>
      <option value="16">Lightning</option>
    </select>
    <span id="r_delta_label" style="font-size:12px;color:#aaa;"></span>
  </div>

  <div style="display:flex;">
    <div id="r_grid"  style="flex:1;height:480px;"></div>
    <div id="r_conv"  style="width:300px;height:480px;"></div>
  </div>

</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(function () {
var _vFrames = %%V_FRAMES%%;
var _pFrames = %%P_FRAMES%%;
var _deltas  = %%DELTAS%%;
var _n       = %%N%%;
var _goal    = %%GOAL%%;       // [display_row, col] after flipud
var _obstacles = %%OBSTACLES%%;  // list of [display_row, col] after flipud
var _max     = %%MAX%%;
var _delay   = %%DELAY%%;

var ARROWS = ['\u2191','\u2193','\u2190','\u2192'];  // ↑↓←→

var _step=0, _playing=false, _timer=null;
var _playBtn = document.getElementById('r_play');
var _slider  = document.getElementById('r_slider');
var _label   = document.getElementById('r_label');
var _dl      = document.getElementById('r_delta_label');
var _speedSel= document.getElementById('r_speed');

var _G = 'rgba(255,255,255,0.09)';

// Build static special-cell markers
function specialMarkers() {
  // Goal star
  var gx = [_goal[1]], gy = [_goal[0]], gt = ['\u2605'];  // ★
  var ox = [], oy = [], ot = [];
  for (var i=0; i<_obstacles.length; i++) {
    ox.push(_obstacles[i][1]); oy.push(_obstacles[i][0]); ot.push('\u2716');  // ✖
  }
  var traces = [];
  if (gx.length) traces.push({
    type:'scatter', x:gx, y:gy, text:gt, mode:'text',
    textfont:{size:22,color:'#fbbf24'}, showlegend:false, hoverinfo:'skip'
  });
  if (ox.length) traces.push({
    type:'scatter', x:ox, y:oy, text:ot, mode:'text',
    textfont:{size:20,color:'#f87171'}, showlegend:false, hoverinfo:'skip'
  });
  return traces;
}

function policyTrace(k) {
  var xs=[], ys=[], texts=[];
  var p = _pFrames[k];
  for (var i=0; i<_n; i++) {
    for (var j=0; j<_n; j++) {
      var act = p[i*_n+j];
      xs.push(j); ys.push(i);
      texts.push(act<0 ? '' : ARROWS[act]);
    }
  }
  return {
    type:'scatter', x:xs, y:ys, text:texts, mode:'text',
    textfont:{size:Math.max(10, Math.floor(240/_n)), color:'white'},
    showlegend:false, hoverinfo:'skip'
  };
}

var _specials = specialMarkers();
var _yTicks   = Array.from({length:_n}, function(_,i){return _n-1-i;});
var _yLabels  = Array.from({length:_n}, function(_,i){return 'r'+i;});

var _layGrid = {
  uirevision:'grid',
  title:{text:'Value Function — sweep 0', font:{size:14,color:'white'}, x:0.5},
  paper_bgcolor:'rgba(17,17,17,1)', plot_bgcolor:'rgba(17,17,17,1)',
  font:{color:'white',family:'monospace'},
  xaxis:{title:'column', gridcolor:_G, zeroline:false, tickvals:Array.from({length:_n},function(_,i){return i;}),
         ticktext:Array.from({length:_n},function(_,i){return 'c'+i;}), range:[-0.5,_n-0.5]},
  yaxis:{title:'row', gridcolor:_G, zeroline:false, tickvals:_yTicks, ticktext:_yLabels,
         range:[-0.5,_n-0.5]},
  height:480, margin:{l:55,r:10,t:45,b:45}
};

var _layConv = {
  uirevision:'conv',
  title:{text:'Max |ΔV| per Sweep', font:{size:13,color:'white'}, x:0.5},
  paper_bgcolor:'rgba(17,17,17,1)', plot_bgcolor:'rgba(17,17,17,1)',
  font:{color:'white',family:'monospace'},
  xaxis:{title:'Sweep',gridcolor:_G,zeroline:false},
  yaxis:{title:'Max |ΔV|',gridcolor:_G,zeroline:false,type:'log'},
  showlegend:false, height:480, margin:{l:60,r:15,t:45,b:45}
};

var _sweeps = Array.from({length:_max}, function(_,i){return i;});

function gridTraces(k) {
  var heatmap = {
    type:'heatmap', z:_vFrames[k],
    x:Array.from({length:_n},function(_,i){return i;}),
    y:Array.from({length:_n},function(_,i){return i;}),
    colorscale:'RdYlGn', showscale:true,
    colorbar:{title:'V(s)',thickness:12,len:0.7},
    hovertemplate:'col:%{x} row:%{y}<br>V: %{z:.4f}<extra></extra>'
  };
  return [heatmap, policyTrace(k)].concat(_specials);
}

function convTraces(k) {
  var allD = _deltas.slice(0,k+1);
  return [
    {type:'scatter', x:_sweeps, y:_deltas, mode:'lines',
     line:{color:'rgba(167,139,250,0.2)',width:1,dash:'dot'}, hoverinfo:'skip'},
    {type:'scatter', x:_sweeps.slice(0,k+1), y:allD, mode:'lines+markers',
     line:{color:'#a78bfa',width:2.5}, marker:{size:4,color:'#a78bfa'},
     hovertemplate:'sweep %{x}<br>Δ %{y:.6f}<extra></extra>'},
    {type:'scatter', x:[k], y:[_deltas[k]], mode:'markers',
     marker:{size:9,color:'white',symbol:'circle'}},
  ];
}

function render(k) {
  _step=k; _slider.value=k;
  _label.textContent = 'Sweep '+k+' / '+(_max-1);
  _dl.textContent    = k>0 ? 'max|ΔV|: '+_deltas[k].toFixed(6) : '';
  _layGrid.title.text = 'Value Function \u2014 sweep '+k;
  Plotly.react('r_grid', gridTraces(k), _layGrid);
  Plotly.react('r_conv', convTraces(k), _layConv);
}

function tick()  { if(_step>=_max-1){pause();return;} render(_step+1); }
function play()  { _playing=true;  _playBtn.textContent='\u23F8 Pause'; _timer=setInterval(tick,_delay); }
function pause() { _playing=false; _playBtn.textContent='\u25B6 Play';  clearInterval(_timer);_timer=null; }
function stop()  { pause(); render(0); }
function togglePlay() { if(_playing) pause(); else play(); }

_playBtn.addEventListener('click', togglePlay);
document.getElementById('r_stop').addEventListener('click', stop);
_slider.addEventListener('input', function(){if(_playing)pause();render(parseInt(this.value,10));});
_speedSel.addEventListener('change', function(){
  _delay=parseInt(this.value,10);
  if(_playing){clearInterval(_timer);_timer=setInterval(tick,_delay);}
});

Plotly.newPlot('r_grid', gridTraces(0), _layGrid, {responsive:true});
Plotly.newPlot('r_conv', convTraces(0), _layConv, {responsive:true});
}());
</script>
</body></html>"""


def _srcdoc(html: str) -> str:
    srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:580px;border:none;background:#111111;border-radius:8px;">'
        f'</iframe>'
    )


def parse_goal(goal_str, n):
    """Parse 'row,col' string into [row, col], clamped to grid."""
    try:
        parts = [int(x.strip()) for x in goal_str.split(",")]
        r, c  = max(0, min(n-1, parts[0])), max(0, min(n-1, parts[1]))
        return [r, c]
    except Exception:
        return [n-1, n-1]


def run_rl(grid_size, goal_str, gamma, n_iter, step_cost, speed):
    n    = int(grid_size)
    goal = parse_goal(goal_str, n)

    # Fixed obstacles that scale with grid size
    obstacles = []
    candidates = [(1,1),(2,3),(3,1),(1,4),(4,2),(2,2),(3,4),(0,3)]
    for r, c in candidates:
        if r < n and c < n and [r, c] != goal:
            obstacles.append([r, c])
        if len(obstacles) >= max(1, n - 2):
            break

    v_frames, p_frames, deltas = value_iteration(
        n, goal, obstacles, gamma, int(n_iter), step_cost
    )

    # Flip goal and obstacle coords for display (np.flipud was applied in v_frames)
    goal_disp = [n - 1 - goal[0], goal[1]]
    obs_disp  = [[n - 1 - r, c] for r, c in obstacles]

    SPEED_MS = {"Slow": 800, "Medium": 300, "Fast": 80, "Lightning": 16}
    sel = {s: "" for s in SPEED_MS}
    sel[speed] = "selected"

    html = (
        _TEMPLATE
        .replace("%%V_FRAMES%%",  json.dumps(v_frames))
        .replace("%%P_FRAMES%%",  json.dumps(p_frames))
        .replace("%%DELTAS%%",    json.dumps(deltas))
        .replace("%%N%%",         str(n))
        .replace("%%GOAL%%",      json.dumps(goal_disp))
        .replace("%%OBSTACLES%%", json.dumps(obs_disp))
        .replace("%%MAX_M1%%",    str(len(v_frames) - 1))
        .replace("%%MAX%%",       str(len(v_frames)))
        .replace("%%DELAY%%",     str(SPEED_MS.get(speed, 300)))
        .replace("%%SEL_MED%%",   sel["Medium"])
        .replace("%%SEL_FAST%%",  sel["Fast"])
    )
    return _srcdoc(html)


GRID_SIZES = [4, 5, 6, 7, 8]

with gr.Blocks(title="RL: Value Iteration — ML Playground") as demo:
    gr.HTML(BACK_HTML)
    gr.Markdown(
        "# RL: Value Iteration\n"
        "### Watch the value function propagate across a grid world sweep by sweep"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            grid_sl  = gr.Slider(4, 8, step=1, value=5, label="Grid Size (N×N)")
            goal_tb  = gr.Textbox(value="4,4", label="Goal Cell (row, col)",
                                  info="0-indexed from top-left")
            gamma_sl = gr.Slider(0.5, 0.999, step=0.001, value=0.95, label="Discount γ")
            cost_sl  = gr.Slider(-0.1, 0.0, step=0.005, value=-0.02, label="Step Cost")
            iter_sl  = gr.Slider(5, 100, step=5, value=40, label="Max Sweeps")
            speed_dd = gr.Dropdown(choices=["Slow","Medium","Fast","Lightning"],
                                   value="Medium", label="Playback Speed")
            run_btn  = gr.Button("▶ Run", variant="primary", size="lg")
            gr.Markdown(
                """
**Legend:** ★ = Goal (+1), ✖ = Obstacle (−1)
Arrows show the greedy policy at each cell.
The log-scale convergence plot shows when V stabilises.
                """
            )

        with gr.Column(scale=2):
            anim_out = gr.HTML(
                "<p style='color:#888;font-family:monospace;padding:20px'>"
                "Configure and click <b>▶ Run</b> to start.</p>"
            )

    run_btn.click(
        fn=run_rl,
        inputs=[grid_sl, goal_tb, gamma_sl, iter_sl, cost_sl, speed_dd],
        outputs=[anim_out],
    )
