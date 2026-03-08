"""
surfaces.py
-----------
Defines the Surface dataclass and all preset loss surfaces.

Each Surface stores:
  f(x, y)          — the scalar loss function
  gradient(x, y)   — analytical gradient (or numerical for custom)
  x_range/y_range  — bounds used for mesh generation and axis limits
  default_start    — sensible (x0, y0) to start optimizers from
  z_clip           — optional ceiling to prevent extreme values blowing
                     out the colour scale (e.g. Rosenbrock hits ~1e5)
  vectorized       — True if f(X, Y) works on numpy 2-D arrays directly.
                     Preset surfaces use element-wise numpy ops, so True.
                     Custom surfaces use eval() on scalars, so False.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional


@dataclass
class Surface:
    name: str
    f: Callable
    gradient: Callable
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    default_start: Tuple[float, float]
    z_clip: Optional[float] = None
    vectorized: bool = True   # can f() accept 2-D numpy arrays?


# Bowl  —  f(x,y) = x² + y²
# Simplest convex surface. Minimum at (0,0). All optimizers converge.

def _bowl_f(x, y):
    return x ** 2 + y ** 2


def _bowl_grad(x, y):
    # ∂f/∂x = 2x,  ∂f/∂y = 2y
    return np.array([2 * x, 2 * y])


# Rosenbrock  —  f(x,y) = (1-x)² + 100(y-x²)²
# The classic "banana valley". Minimum at (1,1) inside a narrow curved
# trough — easy to find the valley, hard to follow it to the bottom.

def _rosenbrock_f(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def _rosenbrock_grad(x, y):
    dfdx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dfdy = 200 * (y - x ** 2)
    return np.array([dfdx, dfdy])


# Himmelblau  —  f(x,y) = (x²+y-11)² + (x+y²-7)²
# Four global minima of equal value. Good for showing how the starting
# point determines which basin an optimizer falls into.

def _himmelblau_f(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def _himmelblau_grad(x, y):
    dfdx = 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)
    dfdy = 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)
    return np.array([dfdx, dfdy])


# Saddle point  —  f(x,y) = x² - y²
# (0,0) is a saddle: a minimum along x but a maximum along y.
# Demonstrates why pure gradient descent can get stuck or escape in
# unintuitive directions.

def _saddle_f(x, y):
    return x ** 2 - y ** 2


def _saddle_grad(x, y):
    return np.array([2 * x, -2 * y])


# Beale  —  f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²
# Minimum at (3, 0.5). Features large flat regions and steep walls —
# a tough test for adaptive-rate methods.

def _beale_f(x, y):
    a = 1.5   - x + x * y
    b = 2.25  - x + x * y ** 2
    c = 2.625 - x + x * y ** 3
    return a ** 2 + b ** 2 + c ** 2


def _beale_grad(x, y):
    a = 1.5   - x + x * y
    b = 2.25  - x + x * y ** 2
    c = 2.625 - x + x * y ** 3
    dfdx = 2 * a * (y - 1)       + 2 * b * (y ** 2 - 1)         + 2 * c * (y ** 3 - 1)
    dfdy = 2 * a * x              + 2 * b * (2 * x * y)           + 2 * c * (3 * x * y ** 2)
    return np.array([dfdx, dfdy])


# Preset surface registry

PRESET_SURFACES = {
    "Bowl (Convex)": Surface(
        name="Bowl",
        f=_bowl_f,
        gradient=_bowl_grad,
        x_range=(-3.0, 3.0),
        y_range=(-3.0, 3.0),
        default_start=(-2.5, 2.5),
        z_clip=20.0,
    ),
    "Rosenbrock (Banana)": Surface(
        name="Rosenbrock",
        f=_rosenbrock_f,
        gradient=_rosenbrock_grad,
        x_range=(-2.0, 2.0),
        y_range=(-1.0, 3.0),
        default_start=(-1.5, 2.0),
        z_clip=500.0,
    ),
    "Himmelblau": Surface(
        name="Himmelblau",
        f=_himmelblau_f,
        gradient=_himmelblau_grad,
        x_range=(-5.0, 5.0),
        y_range=(-5.0, 5.0),
        default_start=(-3.0, 4.0),
        z_clip=300.0,
    ),
    "Saddle Point": Surface(
        name="Saddle",
        f=_saddle_f,
        gradient=_saddle_grad,
        x_range=(-3.0, 3.0),
        y_range=(-3.0, 3.0),
        default_start=(2.0, 0.1),
        z_clip=None,
    ),
    "Beale": Surface(
        name="Beale",
        f=_beale_f,
        gradient=_beale_grad,
        x_range=(-4.5, 4.5),
        y_range=(-4.5, 4.5),
        default_start=(3.0, -3.0),
        z_clip=50.0,
    ),
}


# Custom surface builder

# Whitelist of names available inside custom expressions.
# __builtins__ is blanked so arbitrary Python can't be executed.
_SAFE_MATH = {
    "__builtins__": {},
    "sin": np.sin,  "cos": np.cos,  "tan": np.tan,
    "exp": np.exp,  "log": np.log,  "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "pi": np.pi,    "e": np.e,
}


def make_custom_surface(expr_str: str) -> Surface:
    """
    Build a Surface from a user-supplied expression string in x and y.

    The expression is evaluated with a restricted namespace (no builtins)
    so arbitrary Python can't run. Gradients are computed numerically
    via central finite differences (h=1e-5).

    Raises ValueError if the expression fails at (0, 0).
    """

    def f(x, y):
        # eval with scalar x and y — numpy funcs in _SAFE_MATH handle the math
        env = {**_SAFE_MATH, "x": float(x), "y": float(y)}
        return float(eval(expr_str, env))  # noqa: S307

    def gradient(x, y, h=1e-5):
        # Central finite differences — accurate to O(h²)
        dfdx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        dfdy = (f(x, y + h) - f(x, y - h)) / (2 * h)
        return np.array([dfdx, dfdy])

    # Validate the expression compiles and runs before we hand back the surface
    try:
        f(0.0, 0.0)
    except Exception as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc

    return Surface(
        name="Custom",
        f=f,
        gradient=gradient,
        x_range=(-5.0, 5.0),
        y_range=(-5.0, 5.0),
        default_start=(-3.0, 3.0),
        z_clip=None,
        vectorized=False,   # eval() only works on scalars — needs np.vectorize
    )
