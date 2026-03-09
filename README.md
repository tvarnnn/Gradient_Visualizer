---
title: ML Visualizer
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ML Visualizer

**ML Visualizer** is an interactive machine learning playground built with **FastAPI + Gradio + NumPy + Plotly**.  
It allows users to explore core machine learning concepts through real-time visualizations and simulations.

The app provides multiple interactive modules that demonstrate how different algorithms behave during training.

---

# Modules

## Loss Landscape
Visualize how optimization algorithms move across 2D loss surfaces.

Features:
- 3D loss surface visualization
- Top-down contour view
- Optimizer path animation
- Adjustable learning rate and step count
- Play / pause / scrub training

Optimizers included:
- SGD
- Momentum
- AdaGrad
- RMSProp
- Adam

---

## Decision Boundaries
See how different classifiers partition feature space.

Algorithms:
- KNN
- SVM
- Decision Trees
- Random Forest

Datasets:
- Moons
- Circles
- XOR
- Gaussian blobs

---

## Overfitting vs Generalization
Explore how model complexity affects training and test error.

Features:
- Polynomial regression
- Noise control
- Train vs test loss comparison
- Demonstrates overfitting visually

---

## Clustering
Compare clustering algorithms on synthetic datasets.

Algorithms:
- K-Means
- DBSCAN
- Gaussian Mixture
- Agglomerative Clustering

---

## Momentum Dynamics
Visualize how momentum and adaptive optimizers affect convergence speed and trajectory.

---

## Neural Network Trainer
Train a small neural network and watch the decision boundary evolve in real time.

Features:
- Numpy MLP implementation
- Live loss curve
- Adjustable hidden units and learning rate

---

## Reinforcement Learning
Visualize value iteration on a grid world.

Shows:
- Value function convergence
- Policy improvement
- Greedy action updates

---

# Tech Stack

- **FastAPI**
- **Gradio**
- **NumPy**
- **Plotly**
- **Scikit-Learn**

---
