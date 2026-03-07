"""
Terminal VIX Distribution — CPU Monte Carlo & Visualisation
============================================================

Simulates terminal VIX values under a continuous-time OU stochastic
volatility model augmented with a shot-noise variance jump component.
Paths are generated via Euler–Maruyama discretization on CPU.

The script produces a single side-by-side figure:

    (a) Effect of shot-noise decay rate  b  ∈ {0, 10, 50}
    (b) Effect of initial shot-noise level  L_0 ∈ {0, 1, 5}

Model dynamics (X_t = ln VIX_t):

    dX_t   = κ (m_t − X_t) dt  +  √v_t  dW_t
    dm_t   = κ_m (θ_m − m_t) dt  +  ω_m  dB_t           (OU long-run level)
    dv_t   = κ_1 (θ_1 − v_t) dt  +  ω_1 √v_t dW_t^v    (CIR inst. variance)
             + dL_t^v                                     (shot-noise jumps)
    L_t^v  = Σ_k  J_k  exp(−b (t − τ_k))                (compound Poisson)

    where  τ_k ~ Poisson(η),  J_k ~ Exp(μ_Jv),
    and  Corr(dW_t, dW_t^v) = ρ_1.

Usage
-----
    python numerics_terminal_vix_distribution.py

Author  : FanZ
Created : 2025
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")                    # non-interactive backend: save only
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.stats import gaussian_kde
from tqdm import tqdm

# Optional: use SciencePlots style if installed
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass

# ---------------------------------------------------------------------------
#  Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================== #
#  1.  MODEL & SIMULATION CONFIGURATION                                 #
# ===================================================================== #

@dataclass
class ModelParams:
    """Calibrated parameters for the CTOU-SV shot-noise VIX model.

    All values are obtained from maximum-likelihood estimation on
    historical VIX data.  See Table 1 in the paper for details.
    """

    # -- Market / maturity ------------------------------------------------
    r: float = 0.03              # risk-free rate
    T: float = 90 / 360          # option maturity (90 calendar days)

    # -- VIX mean-reversion -----------------------------------------------
    kappa: float = 5.0898        # speed of mean reversion for X_t

    # -- Long-run level m_t  (OU process) ---------------------------------
    kappa_m: float = 1.0218      # mean-reversion speed for m_t
    theta_m: float = 3.4247      # long-run mean of m_t
    omega_m: float = 0.2282      # volatility of m_t

    # -- Instantaneous variance v_t  (CIR-type) --------------------------
    kappa_1: float = 2.4754      # mean-reversion speed for v_t
    theta_1: float = 2.5097      # long-run mean of v_t
    omega_1: float = 1.1062      # vol-of-vol  (diffusion coeff. of v_t)
    rho_1: float = 0.4976        # correlation  Corr(dW, dW^v)

    # -- Shot-noise component L_t^v ---------------------------------------
    b: float = 12.3813           # exponential decay rate of jump impact
    eta: float = 4.8194          # Poisson jump intensity
    mu_Jv: float = 1.5831        # mean of exponential jump-size dist.

    # -- Initial conditions -----------------------------------------------
    VIX_0: float = 20.0          # initial VIX level
    v1_0: float = 0.64           # initial instantaneous variance
    l0: float = 0.0              # initial shot-noise level  L_0
    m_0: float = 3.0             # initial long-run level  m_0


@dataclass
class SimConfig:
    """Monte Carlo simulation settings."""
    n_paths: int = 200000       # number of independent MC paths
    n_steps: int = 128           # Euler-Maruyama time steps per path
    seed: int = 10               # NumPy RNG seed


# ===================================================================== #
#  2.  MONTE CARLO ENGINE                                               #
# ===================================================================== #

def simulate_terminal_vix(
    params: ModelParams,
    config: SimConfig,
    *,
    b_override: float | None = None,
    l0_override: float | None = None,
) -> np.ndarray:
    """Run an Euler-Maruyama simulation and return terminal VIX values.

    Parameters
    ----------
    params : ModelParams
        Calibrated model parameters.
    config : SimConfig
        Monte Carlo simulation settings.
    b_override : float, optional
        If supplied, use this value of *b* instead of ``params.b``.
    l0_override : float, optional
        If supplied, use this value of *L_0* instead of ``params.l0``.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Terminal VIX values  VIX_T = exp(X_T)  at maturity.
    """
    dt = params.T / config.n_steps
    b = b_override if b_override is not None else params.b
    l0_init = l0_override if l0_override is not None else params.l0

    terminal = np.empty(config.n_paths)

    for i in tqdm(range(config.n_paths), desc="  MC paths", ncols=80):
        # Initialise state variables
        X  = np.log(params.VIX_0)        # log-VIX
        V1 = params.v1_0                 # instantaneous variance
        m  = params.m_0                  # long-run level
        Lv = l0_init                     # shot-noise component

        for _ in range(config.n_steps):
            # --- Correlated Brownian increments -----------------------
            Z1 = np.random.standard_normal()          # drives X_t
            Z3 = np.random.standard_normal()          # auxiliary
            Z_v = (params.rho_1 * Z1                  # drives v_t
                   + np.sqrt(1.0 - params.rho_1 ** 2) * Z3)
            B = np.random.standard_normal()            # drives m_t

            # --- Poisson jump count via thinning method --------------
            n_jumps = 0
            if params.eta > 0:
                cum, lam = 0.0, params.eta * dt
                while cum < 1.0:
                    cum -= np.log(np.random.uniform() + 1e-10) / lam
                    n_jumps += 1
                n_jumps -= 1   # the last arrival exceeded the interval

            # --- Exponential jump size  J ~ Exp(1/mu_Jv) -------------
            J_var = -params.mu_Jv * np.log(np.random.uniform() + 1e-16)

            # --- Save previous state for explicit Euler step ---------
            X_prev, V1_prev, m_prev, Lv_prev = X, V1, m, Lv

            # --- Update shot-noise  L_t (deterministic decay + jumps)
            Lv = max(Lv - b * Lv * dt + J_var * n_jumps, 1e-8)
            dLv = Lv - Lv_prev

            # --- Update long-run level  m_t  (OU process) ------------
            m = max(m + params.kappa_m * (params.theta_m - m) * dt
                    + params.omega_m * np.sqrt(dt) * B, 1e-8)

            # --- Update inst. variance  v_t  (CIR + shot-noise) ------
            V1 = max(V1 + params.kappa_1 * (params.theta_1 - V1) * dt
                     + params.omega_1 * np.sqrt(V1) * np.sqrt(dt) * Z_v
                     + dLv, 1e-8)

            # --- Update log-VIX  X_t  (mean-reverting diffusion) -----
            X = X_prev + params.kappa * (m_prev - X_prev) * dt \
                + np.sqrt(V1_prev) * np.sqrt(dt) * Z1

        # Store terminal VIX value
        terminal[i] = np.exp(X)

    return terminal


# ===================================================================== #
#  3.  SCENARIO RUNNERS                                                 #
# ===================================================================== #

def run_scenarios(
    params: ModelParams,
    config: SimConfig,
    values: List[float],
    mode: str = "b",
) -> Dict[str, np.ndarray]:
    """Run Monte Carlo for each value in *values* and collect results.

    Parameters
    ----------
    params : ModelParams
        Base model parameters (the varied parameter is overridden).
    config : SimConfig
        Simulation settings shared across all scenarios.
    values : list of float
        Parameter values to sweep.
    mode : {"b", "l0"}
        ``"b"``  -- sweep the shot-noise decay rate *b*.
        ``"l0"`` -- sweep the initial shot-noise level *L_0*.

    Returns
    -------
    dict
        ``{label: terminal_vix_array}`` for each scenario.
    """
    distributions: Dict[str, np.ndarray] = {}
    for val in values:
        label = f"b={val:.1f}" if mode == "b" else f"l0={val:.1f}"
        logger.info("Scenario %-14s  (%s paths)", label, f"{config.n_paths:,}")
        kw = {"b_override": val} if mode == "b" else {"l0_override": val}
        distributions[label] = simulate_terminal_vix(params, config, **kw)
    return distributions


# ===================================================================== #
#  4.  VISUALISATION                                                    #
# ===================================================================== #

# Colour palette (ColorBrewer Set1) and line styles for <= 3 scenarios
COLORS = ["#E41A1C", "#377EB8", "#4DAF4A"]   # red, blue, green
STYLES = ["-", "--", "dotted"]


def _plot_panel(
    ax: plt.Axes,
    distributions: Dict[str, np.ndarray],
    title: str,
    *,
    inset_range: Tuple[float, float] = (65, 75),
    inset_borderpad: float = 8.5,
) -> None:
    """Draw Gaussian-KDE density curves on *ax* with a zoomed inset.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes for the main density plot.
    distributions : dict
        ``{label: array}`` -- each array is a 1-D sample of terminal VIX.
    title : str
        Panel title, e.g. ``"(a) Effects of $b$"``.
    inset_range : (lo, hi)
        x-range for the zoomed-in right-tail inset.
    inset_borderpad : float
        Padding between inset and axes border.
    """
    # --- Main KDE density curves -----------------------------------------
    for (label, data), c, ls in zip(distributions.items(), COLORS, STYLES):
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), 80, 300)
        ax.plot(x, kde(x), color=c, lw=2.5, alpha=0.9, ls=ls, label=label)

    # --- Axis formatting -------------------------------------------------
    ax.set(xlabel="Terminal VIX Value", ylabel="Density", xlim=(0, 80))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.label.set(fontsize=13, fontweight="bold")
    ax.yaxis.label.set(fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right", frameon=True,
              fancybox=False, edgecolor="gray", framealpha=0.95)
    ax.grid(True, alpha=0.25, ls="--", lw=0.8)
    ax.set_facecolor("#f8f9fa")

    # --- Zoomed inset on the right tail ----------------------------------
    lo, hi = inset_range
    ax_ins = inset_axes(ax, width="25%", height="35%",
                        loc="lower right", borderpad=inset_borderpad)
    for (_, data), c, ls in zip(distributions.items(), COLORS, STYLES):
        kde = gaussian_kde(data)
        x = np.linspace(lo, hi, 300)
        ax_ins.plot(x, kde(x), color=c, lw=3, alpha=0.9, ls=ls)
    ax_ins.set_xlim(lo, hi)
    ax_ins.set_xticks([lo, (lo + hi) / 2, hi])
    ax_ins.set_yticks([])
    ax_ins.set_facecolor("#fafafa")
    ax_ins.grid(True, alpha=0.25, ls="--", lw=0.8)
    mark_inset(ax, ax_ins, loc1=1, loc2=3,
               fc="none", ec="0.6", lw=1.2, ls="--")


def plot_terminal_distributions(
    dist_b: Dict[str, np.ndarray],
    dist_l0: Dict[str, np.ndarray],
    save_path: str | Path = "terminal_vix_combined_distribution.png",
    dpi: int = 400,
) -> None:
    """Create and save the side-by-side distribution figure.

    Left panel  -- effect of decay rate *b*.
    Right panel -- effect of initial shot-noise level *L_0*.
    Both panels include a zoomed inset highlighting the right tail.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _plot_panel(ax1, dist_b,  r"(a) Effects of $b$")
    _plot_panel(ax2, dist_l0, r"(b) Effects of $L_0$")

    fig.suptitle(r"Distribution of $VIX_T$",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved -> %s", save_path)

    # Open the saved image with the system default viewer
    import os, subprocess, sys
    abs_path = str(Path(save_path).resolve())
    if sys.platform == "win32":
        os.startfile(abs_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", abs_path])
    else:
        subprocess.run(["xdg-open", abs_path])


# ===================================================================== #
#  5.  ENTRY POINT                                                      #
# ===================================================================== #

# -- Simulation settings (edit here) -------------------------------------
N_PATHS = 200000       # MC paths per scenario
N_STEPS = 128           # Euler-Maruyama time steps per path
SEED    = 10            # RNG seed for b-scenarios  (l0 uses seed=42)
# Output figure saved next to this script
OUTPUT  = str(Path(__file__).resolve().parent / "terminal_vix_combined_distribution.png")
# -------------------------------------------------------------------------


def main() -> None:
    """Entry point: simulate six scenarios and produce the figure."""
    params = ModelParams()
    config = SimConfig(n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED)

    # ---- (a) Sweep decay rate  b in {0, 10, 50} ------------------------
    np.random.seed(config.seed)
    t0 = time.perf_counter()
    dist_b = run_scenarios(params, config,
                           values=[0.0, 10.0, 50.0], mode="b")
    logger.info("b scenarios done in %.1f s", time.perf_counter() - t0)

    # ---- (b) Sweep initial level  L_0 in {0, 1, 5} ---------------------
    np.random.seed(42)                   # independent seed
    t0 = time.perf_counter()
    dist_l0 = run_scenarios(params, config,
                            values=[0.0, 1.0, 5.0], mode="l0")
    logger.info("l0 scenarios done in %.1f s", time.perf_counter() - t0)

    # ---- Produce and save figure ----------------------------------------
    plot_terminal_distributions(dist_b, dist_l0, save_path=OUTPUT)
    logger.info("Done.")


if __name__ == "__main__":
    main()
