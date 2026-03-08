# Gradience

**Gradience** is an interactive machine learning optimization visualizer built with **Python, Gradio, NumPy, and Plotly**. It lets users explore how gradient-based optimizers move across 2D loss landscapes in real time.

The app renders a **3D loss surface**, a **top-down contour view**, and a **loss-vs-step chart**, then animates optimizer trajectories step-by-step so you can compare convergence behavior visually.

This project was built as an educational and exploratory tool for understanding **optimization dynamics in machine learning**.

---

## Features

- Interactive visualization of **loss landscapes** in 3D
- Animated optimizer trajectories across the surface
- Side-by-side comparison of multiple optimizers
- Top-down **contour view** for clearer path tracking
- **Loss vs. step** chart for convergence analysis
- Adjustable:
  - learning rate
  - starting point
  - number of steps
  - playback speed
- Support for **custom user-defined surfaces**
- Client-side playback controls:
  - play
  - pause
  - stop
  - scrub slider
- Camera position is preserved during animation updates

---

## Implemented Optimizers

- **SGD**
- **Momentum**
- **AdaGrad**
- **RMSProp**
- **Adam**

Each optimizer is implemented manually in **pure NumPy**.

---

## Implemented Loss Surfaces

- **Bowl (Convex)**
- **Rosenbrock (Banana)**
- **Himmelblau**
- **Saddle Point**
- **Beale**
- **Custom function input**

Preset surfaces use analytical gradients.  
Custom surfaces use **numerical gradients via central finite differences**.

---

## How It Works

The program simplifies optimization to a 2-parameter setting, where the loss function is:

```math
z = f(x, y)
