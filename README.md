# VIX Spike — Vol-of-Vol Model: Replication Package

<!-- Badges -->
[![PyPI](https://img.shields.io/pypi/v/VIXDirectSpike?label=PyPI)](https://pypi.org/project/VIX-Spike/)
[![License](https://img.shields.io/pypi/l/VIXDirectSpike?label=License)](https://pypi.org/project/VIX-Spike/)
[![GitHub repo](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/Zheqi-Fan/VIXDirectSpike)
[![Downloads/month](https://img.shields.io/pypi/dm/VIXDirectSpike?label=Downloads)](https://pypi.org/project/VIX-Spike/)

> **Replication package for:**
>
> Fan, Z., Ryu, D., & Ye, Y. (2026). *Valuation of VIX derivatives: Incorporating larger spikes in volatility-of-volatility dynamics.*

This repository contains the full replication code for the above paper. The project studies VIX option pricing under a continuous-time stochastic volatility model augmented with shot-noise variance jumps (compound Poisson process with exponentially decaying magnitudes). Three model specifications are studied:

| Label | Description |
|---|---|
| **CTOUSV** | Baseline: Poisson-driven stochastic vol-of-vol |
| **CTOUSV+** | Brownian motion-driven stochastic vol-of-vol |
| **CTOUSV++** | Our proposed model |

---

## Repository Structure

```
VIXSpike Codess/
├── README.md
│
├── empirics_dm_statistic.py          # Diebold-Mariano test (model comparison)
├── empirics_ecdf.py                  # ECDF / KDE plots of pricing errors
├── empirics_estimate.py              # two-step calibration framework (VERY time-consuming!)
├── empirics_filtered_latent_var.py   # Time-series plots of latent state vars
│
├── numerics_cuda_role.py             # CUDA Monte Carlo & sensitivity analysis
├── numerics_shot_noise_visualization.py  # Shot noise process visualization
├── numerics_terminal_vix_distribution.py # Terminal VIX distribution (CPU MC)
│
├── utils2.py                         # Black76 pricing utilities
│
├── Data/
│   ├── FinalFilteredData0717-monthly.csv  # VIX option panel (2007–2017)
│   ├── VIXSpot0722.csv                    # VIX spot index (2007–2022)
│   └── Yahoo VVIX.csv                     # VVIX index (vol-of-vol)
│
├── Results/
│   ├── fitted_value_CTOUSV*.csv           # In-sample/OOS fitted prices
│   ├── latentVar_df_*.csv                 # Filtered latent variables
│   └── *.png                              # Generated figures
│
└── Pricer/                           # Modular FFT+MC pricing library
    ├── model_config.py
    ├── characteristic_function.py
    ├── fft_pricing.py
    ├── monte_carlo.py
    ├── validator.py
    └── run_validation.py
```

---

## File Descriptions

### Empirics Scripts (root level)

#### `empirics_dm_statistic.py`
Performs the **Diebold-Mariano (DM) test** to compare predictive pricing accuracy across the three model specifications. The benchmark is CTOUSV++ (most general model); nested models (CTOUSV, CTOUSV+) are tested as alternatives.

- Loss differential: vega-weighted absolute pricing error
- Standard errors: Newey-West HAC with 6 lags
- Tests conducted separately for in-sample (pre-2014) and out-of-sample periods
- **Inputs:** `Data/FinalFilteredData0717-monthly.csv`, `Results/fitted_value_*.csv`
- **Output:** DM statistics and p-values printed to console

#### `empirics_ecdf.py`
Plots the **empirical CDF and kernel density** of vega-weighted pricing errors for all three model specifications.

- Produces two figures (in-sample and out-of-sample), each with an ECDF panel (left) and a histogram + KDE panel (right)
- Errors are winsorized at the 1st/99th percentile before plotting
- **Inputs:** `Data/FinalFilteredData0717-monthly.csv`, `Results/fitted_value_*.csv`
- **Outputs:** `Results/histogram_density_ins-VolOfVol.png`, `Results/histogram_density_oos-VolOfVol.png`

#### `empirics_estimate.py`
Performs **two-step calibration** of the baseline CTOUSV model (OU long-run level $m_t$, no jump) to VIX option data.

- **Step 1 (per date):** Brute-force grid search over latent state variables ($V_t$, $m_t$).
- **Step 2 (global):** Differential-evolution optimisation of structural parameters over the full in-sample panel.
- Repeats the two-step loop for `N_ITER` iterations; applies stratified sampling (≤ 100 observations per date).
- In-sample cutoff: dates before `2014-13`; OOS Step 1 only (structural parameters frozen at in-sample estimates).
- **Dependencies:** `tensorflow` / `keras`, `scikit-learn`, `scipy`, `tqdm`, `scienceplots`
- **Inputs:** `Data/FinalFilteredData0717-monthly.csv`, `Data/Yahoo VVIX.csv`, `Results/latentVar_df_SVJ-vega.csv`
- **Outputs:** `Results/latentVar_df_CTOUSV-vega.csv`, `Results/latentVarOOS_df_CTOUSV.csv`, `Results/fitted_value_CTOUSV.csv`, `Results/fitted_value_CTOUSV_OOS.csv`, `Results/latent_CTOUSV_InS.png`

#### `empirics_filtered_latent_var.py`
Generates **time-series plots of filtered latent state variables** ($m_t$ and $V_t$) alongside log-VIX, for all three model specifications.

- Produces two figures: one for the long-run level $m_t$ (1×3 panel) and one for instantaneous variance $V_t$ (1×3 panel)
- Shaded areas distinguish the out-of-sample period; colored lines use a consistent per-model palette
- **Inputs:** `Results/latentVar_df_*.csv`, `Results/latentVarOOS_df_*.csv`, `Data/VIXSpot0722.csv`
- **Outputs:** `Results/latent_mt-VolOfVol.png`, `Results/latent_Vt-VolOfVol.png`

---

### Numerics Scripts (root level)

#### `numerics_cuda_role.py`
**CUDA-accelerated Monte Carlo** pricing and sensitivity analysis for VIX call options under the shot-noise model.

- Implements a custom CUDA kernel (via Numba) using Euler-Maruyama discretization (512 steps, 1,000,000 paths)
- Computes Black76 implied volatilities from Monte Carlo prices
- Runs a comprehensive parameter sensitivity sweep across $\kappa$, $\eta$, $\mu_{J_v}$, $b_v$, $\rho_1$
- Includes a sanity-check comparing FFT vs Monte Carlo prices
- **Dependencies:** `numba`, `numba.cuda`, `scipy`, `scienceplots`
- **Output:** Sensitivity analysis figures saved to `Results/`

#### `numerics_shot_noise_visualization.py`
Visualizes **shot noise sample paths** with different jump distributions and decay rates.

- Side-by-side comparison: Normal jump distribution (left panel) vs. Exponential jump distribution (right panel)
- Varies mean-reversion decay rate $b$ to show its effect on trajectory shape
- Uses Euler-Maruyama simulation for the compound Poisson shot noise process
- **Output:** Figure saved to `Results/`

#### `numerics_terminal_vix_distribution.py`
Simulates the **terminal VIX distribution** via CPU-based Euler-Maruyama Monte Carlo under the full CTOUSV++ dynamics.

- Produces a side-by-side figure with two panels:
  - (a) Effect of shot-noise decay rate $b \in \{0, 10, 50\}$
  - (b) Effect of initial shot-noise level $L_0 \in \{0, 1, 5\}$
- Includes kernel density estimation (KDE) with zoomed inset axes
- **Dependencies:** `scipy`, `tqdm`, `scienceplots`
- **Output:** Figure saved to `Results/`

---

### Utilities

#### `utils2.py`
Shared **Black76 pricing utilities** used across multiple scripts.

| Function | Description |
|---|---|
| `Black76_Call` | European call price on a futures contract (Black 1976) |
| `Black76_Put`  | European put price on a futures contract (Black 1976) |
| `Black76_IV`   | Implied volatility via bisection (verified against MATLAB `blkimpv`) |

---

### `Pricer/` — Modular VIX Option Pricing Library

A self-contained, modular implementation of the model (variance jumps only). The FFT and Monte Carlo pricers can be imported independently.

#### `Pricer/model_config.py`
Defines **data structures** (`dataclass`) for all model inputs:
- `ModelParameters` — 12 model parameters ($\kappa$, $\kappa_m$, $\theta_m$, $\omega_m$, $\kappa_1$, $\theta_1$, $\omega_1$, $\rho_1$, $b_v$, $\lambda$, $\mu_{J_V}$, $T$)
- `InitialConditions` — initial values of state variables ($\text{VIX}_0$, $v_{1,0}$, $L_0^v$, $m_0$); note $\text{VIX}_0$ is stored in levels (not log-transformed)

#### `Pricer/characteristic_function.py`
Solves the **characteristic function** of $\log(\text{VIX}_T)$ via a system of coupled Riccati-type ODEs for the affine coefficients $A$, $B$, $D$, $E$, $F$.

- Coefficient $A$ is solved analytically; $B$, $D$, $E$, $F$ are obtained by numerical RK4 ODE integration
- Provides `compute_characteristic_function(u, T, params, initial)` callable

#### `Pricer/fft_pricing.py`
Implements **Carr-Madan FFT pricing** of European VIX call options.

- `FFTPricer.price_call_fft()` — prices a full array of strikes using FFT
- `FFTPricer.price_single_call()` — interpolates to a single target strike
- Configurable damping parameter $\alpha$, frequency grid spacing $\eta$, FFT size $N = 2^n$

#### `Pricer/monte_carlo.py`
**Euler-Maruyama Monte Carlo** path simulator for the model.

- `MonteCarloSimulator.generate_paths()` — simulates paths of $X_t$ (log-VIX), $m_t$, $v_t$, $L_t^v$
- `MonteCarloSimulator.price_call_mc()` — computes call prices from terminal VIX paths
- Applies reflection at zero boundary for variance processes

#### `Pricer/validator.py`
**Validation framework** that compares FFT and Monte Carlo prices across a grid of strikes.

- `ModelValidator.validate_pricing()` — runs both pricers and returns a comparison `DataFrame` (Strike, FFT Price, MC Price, Absolute Error, Relative Error %)

#### `Pricer/run_validation.py`
Main **entry-point script** for the `Pricer/` module. Sets representative parameter values, runs the validation, and prints a formatted results table.

```bash
cd Pricer
python run_validation.py
```

---

## Data Files

| File | Description |
|---|---|
| `Data/FinalFilteredData0717-monthly.csv` | Filtered VIX option panel (monthly, 2007–2017); columns include strike, maturity, mid-price, vega |
| `Data/VIXSpot0722.csv` | Daily VIX spot index (CBOE), 2007–2022 |
| `Data/Yahoo VVIX.csv` | Daily VVIX index (implied vol-of-vol), sourced from Yahoo Finance |

---

## Results Files

| File | Description |
|---|---|
| `Results/fitted_value_CTOUSV[P][PP][_OOS].csv` | Model-fitted option prices for each specification (in-sample and out-of-sample) |
| `Results/latentVar_df_*-vega.csv` | In-sample filtered latent state variables ($m_t$, $V_t$, etc.) |
| `Results/latentVarOOS_df_*.csv` | Out-of-sample filtered latent state variables |
| `Results/latentVar_df_6.csv` | Auxiliary latent variable output |

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
seaborn
scienceplots
statsmodels
numba          # for CUDA Monte Carlo (requires CUDA-capable GPU)
tqdm
tensorflow     # for empirics_estimate_CTOUSV.py
scikit-learn   # for empirics_estimate_CTOUSV.py
```

Install all dependencies via:

```bash
pip install numpy pandas scipy matplotlib seaborn scienceplots statsmodels numba tqdm tensorflow scikit-learn
```

> **Note:** `numerics_cuda_role.py` requires a CUDA-capable NVIDIA GPU and the CUDA toolkit. All other scripts run on CPU only.

---

## Usage

Run any top-level script from the project root:

```bash
python empirics_dm_statistic.py
python empirics_ecdf.py
python empirics_estimate_CTOUSV.py
python empirics_filtered_latent_var.py
python numerics_terminal_vix_distribution.py
python numerics_shot_noise_visualization.py
python numerics_cuda_role.py          # requires GPU
```

Run the modular pricer validation:

```bash
cd Pricer
python run_validation.py
```

---

## References

Fan, Z., Ryu, D., & Ye, Y. (2026). *Valuation of VIX derivatives: Incorporating larger spikes in volatility-of-volatility dynamics.*




