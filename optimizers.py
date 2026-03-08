"""
Pure-NumPy implementations of five gradient-based optimizers.

All optimizers share the same simple interface:
    opt = SomeOptimizer(lr)
    opt.reset()                     # clear internal state (call before each run)
    new_params = opt.step(params, grads)   # return updated parameter vector

`params` and `grads` are both 1-D NumPy arrays of shape (2,)
representing the (x, y) position in parameter space.

Optimizers included (in historical/conceptual order):
    SGD        → baseline
    Momentum   → adds velocity to smooth oscillations
    AdaGrad    → per-parameter adaptive learning rate
    RMSProp    → fixes AdaGrad's vanishing rate problem
    Adam       → Momentum + RMSProp with bias correction (modern default)
"""

import numpy as np


# SGD — Stochastic Gradient Descent

class SGD:
    """
    The simplest optimizer: just subtract lr * gradient at every step.

    Has no internal state, so `reset()` is a no-op.
    Very sensitive to the choice of learning rate.
    """
    name = "SGD"

    def __init__(self, lr):
        self.lr = lr

    def reset(self):
        pass  # no internal state to clear

    def step(self, params, grads):
        # θ ← θ - α∇f(θ)
        return params - self.lr * grads


# Momentum

class Momentum:
    """
    Adds a "velocity" term that accumulates gradient history.

    v ← β·v + ∇f(θ)
    θ ← θ - α·v

    This smooths out oscillations (especially in ravines) and can
    accelerate convergence along consistent gradient directions.
    β = 0.9 is the typical default.
    """
    name = "Momentum"

    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta  # momentum decay factor (0 = no memory, 1 = infinite memory)
        self.v = None     # velocity vector; initialised lazily on first step

    def reset(self):
        self.v = None  # forget accumulated velocity between runs

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)  # cold-start at zero velocity
        # Roll the velocity forward and apply
        self.v = self.beta * self.v + grads
        return params - self.lr * self.v


# AdaGrad — Adaptive Gradient

class AdaGrad:
    """
    Adapts the learning rate per parameter by dividing by the
    square root of the sum of all past squared gradients.

    G ← G + ∇f²
    θ ← θ - α / (√G + ε) · ∇f

    Parameters that receive large gradients get a smaller effective lr,
    and vice-versa. Great for sparse data, but the ever-growing G means
    the learning rate can shrink to almost zero on long runs.
    """
    name = "AdaGrad"

    def __init__(self, lr, eps=1e-8):
        self.lr = lr
        self.eps = eps   # small constant to prevent division by zero
        self.G = None    # accumulated squared gradients (one value per param)

    def reset(self):
        self.G = None

    def step(self, params, grads):
        if self.G is None:
            self.G = np.zeros_like(params)
        self.G += grads ** 2                                    # accumulate squared grads
        return params - self.lr * grads / (np.sqrt(self.G) + self.eps)


# RMSProp — Root Mean Square Propagation

class RMSProp:
    """
    Fixes AdaGrad's vanishing learning-rate problem by using an
    exponential moving average of squared gradients instead of a
    running sum.

    v ← decay·v + (1 - decay)·∇f²
    θ ← θ - α / (√v + ε) · ∇f

    The EMA keeps the denominator from growing without bound, making
    RMSProp much better suited to non-stationary problems (e.g. RNNs).
    decay = 0.9 is the typical default (Hinton's original suggestion).
    """
    name = "RMSProp"

    def __init__(self, lr, decay=0.9, eps=1e-8):
        self.lr = lr
        self.decay = decay  # EMA decay rate for squared gradients
        self.eps = eps
        self.v = None       # EMA of squared gradients

    def reset(self):
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        # Exponential moving average of squared gradients
        self.v = self.decay * self.v + (1 - self.decay) * grads ** 2
        return params - self.lr * grads / (np.sqrt(self.v) + self.eps)


# Adam — Adaptive Moment Estimation

class Adam:
    """
    Combines Momentum (1st moment) and RMSProp (2nd moment), with
    bias-correction terms that counteract the zero-initialisation of
    m and v at the start of training.

    m ← β1·m + (1 - β1)·∇f          # 1st moment (mean)
    v ← β2·v + (1 - β2)·∇f²         # 2nd moment (uncentred variance)
    m̂ = m / (1 - β1^t)               # bias-corrected mean
    v̂ = v / (1 - β2^t)               # bias-corrected variance
    θ ← θ - α · m̂ / (√v̂ + ε)

    Default hyper-parameters (β1=0.9, β2=0.999, ε=1e-8) work well
    across most deep learning tasks and rarely need tuning.
    """
    name = "Adam"

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1   # decay for 1st moment (momentum)
        self.beta2 = beta2   # decay for 2nd moment (RMSProp-style)
        self.eps = eps
        self.m = None        # 1st moment vector
        self.v = None        # 2nd moment vector
        self.t = 0           # step counter (used for bias correction)

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0  # reset step counter so bias correction starts fresh

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        # Update biased moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        # Bias-corrected estimates
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# Registry — maps string names to classes

# Used by visualization.py to look up the right class by name
OPTIMIZERS = {
    "SGD": SGD,
    "Momentum": Momentum,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
    "Adam": Adam,
}

# Colour assigned to each optimizer in the 3D plot
OPTIMIZER_COLORS = {
    "SGD": "#e74c3c",        # red
    "Momentum": "#e67e22",   # orange
    "AdaGrad": "#2ecc71",    # green
    "RMSProp": "#3498db",    # blue
    "Adam": "#9b59b6",       # purple
}
