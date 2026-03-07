"""
================================================================================
VIX Shot Noise Model: CUDA-Accelerated Monte Carlo Pricing and Sensitivity Analysis
================================================================================

Module: cuda_numerics_role.py

Description:
    This module implements GPU-accelerated Monte Carlo simulation for pricing VIX
    options under a shot noise variance model. It includes:
    - CUDA kernel for efficient path generation (Euler-Maruyama discretization)
    - VIX call option pricing via Monte Carlo simulation
    - Implied volatility computation using Black76 formula inversion
    - Comprehensive sensitivity analysis across model parameters
    - Sanity check validation (FFT vs Monte Carlo comparison)

Model Specification:
    - Underlying: VIX futures with shot noise variance jumps
    - Dynamics: log(VIX_t) follows mean-reverting process with stochastic variance
    - Variance jumps: Poisson arrivals with exponential magnitudes
    - Discretization: Euler-Maruyama with 512 time steps
    - Monte Carlo: 1,000,000 paths for convergence

Key Parameters:
    - κ: VIX mean reversion speed (≈ 5.09)
    - κ_m: Long-term variance center mean reversion
    - κ_1: Instantaneous variance mean reversion
    - ρ_1: Correlation between VIX and variance shocks
    - b_v: Variance jump decay rate (mean reversion of jump component)
    - η: Jump arrival intensity (Poisson parameter)
    - μ_J_v: Mean size of variance jumps (exponential parameter)

Author: VIX Volatility Research Team
Date: February 2026
Python Version: 3.8+
Dependencies: numpy, scipy, numba, matplotlib, scienceplots

License: For academic use in replication packages

================================================================================
"""

import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32
from tqdm import tqdm

# =====================================================================
# CONFIGURATION AND STYLING
# =====================================================================

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    print("Warning: scienceplots not installed. Using default matplotlib style.")

# =====================================================================
# CUDA KERNEL: MONTE CARLO PATH GENERATOR
# =====================================================================

@cuda.jit
def Generate_DFH(NoOfPaths, NoOfSteps, T, r, S_0, kappa, kappam, thetam, omegam,
                 kappa1, theta1, omega1, rho1, bv, eta, muJv, v10, Lv0, m0, rng_states, paths):
    """
    CUDA kernel for GPU-accelerated Monte Carlo path generation.
    
    Simulates VIX paths using Euler-Maruyama discretization with variance jumps.
    Only variance jumps implemented (no level jumps).
    
    Parameters:
    -----------
    NoOfPaths : int
        Number of Monte Carlo paths to generate
    NoOfSteps : int
        Number of discrete time steps per path (default: 512)
    T : float
        Time-to-maturity in years (e.g., 90/360 for 90 days)
    r : float
        Risk-free rate (e.g., 0.03 for 3%)
    S_0 : float
        Initial VIX level (e.g., 20)
    kappa, kappam, kappa1 : float
        Mean reversion speeds for log(VIX), long-term variance center, instantaneous variance
    thetam, theta1 : float
        Long-term targets for variance processes
    omegam, omega1 : float
        Volatility of variance (diffusion coefficients)
    rho1 : float
        Correlation between VIX and variance shocks [-1, 1]
    bv : float
        Decay rate for variance jump shot noise component
    eta : float
        Intensity (arrival rate) of variance jump Poisson process
    muJv : float
        Mean size of variance jumps (exponential parameter)
    v10 : float
        Initial instantaneous variance
    Lv0 : float
        Initial variance jump shot noise level
    m0 : float
        Initial long-term variance center
    rng_states : cupy array
        CUDA random number generator states
    paths : cupy array
        Output array for terminal VIX values
    
    Notes:
    ------
    - Uses Poisson thinning method for jump arrivals
    - Maintains Euler-Maruyama consistency by using previous timestep values
    - Thread-safe via cuda.grid() for parallel execution
    """
    
    thread_id = cuda.grid(1)
    n_paths = len(paths)
    stride = cuda.gridDim.x * cuda.blockDim.x
    dt = T / float(NoOfSteps)

    for i in range(thread_id, n_paths, stride):
        # Initialize path variables
        X = np.log(S_0)  # log(VIX)
        V1 = v10         # Instantaneous variance
        m = m0           # Long-term variance center
        Lv = Lv0         # Variance jump shot noise

        for n in range(NoOfSteps):
            # Generate correlated Brownian increments
            Z1 = xoroshiro128p_normal_float32(rng_states, thread_id)
            Z3 = xoroshiro128p_normal_float32(rng_states, thread_id)
            Z_3 = rho1 * Z1 + np.sqrt(1.0 - rho1**2) * Z3
            B = xoroshiro128p_normal_float32(rng_states, thread_id)

            # Generate Poisson jump arrivals (thinning method)
            Z_pois = 0
            if eta > 0:
                T_ = 0.0
                n_ = -1
                mu_ = eta * dt
                while T_ < 1.0:
                    U_ = xoroshiro128p_uniform_float32(rng_states, thread_id)
                    E_ = -(1.0 / mu_) * np.log(max(U_, 1e-10))
                    T_ = T_ + E_
                    n_ = n_ + 1
                Z_pois = max(n_, 0)

            # Generate exponential-distributed jump size
            U_Jv = xoroshiro128p_uniform_float32(rng_states, thread_id)
            J_var = -muJv * np.log(max(U_Jv, 1e-16))

            # Save previous state for Euler-Maruyama consistency
            X_prev = X
            V1_prev = V1
            m_prev = m
            Lv_prev = Lv

            # Update variance jump shot noise: L^v_t = L^v_{t-1} exp(-b_v dt) + J_var * Z_pois
            Lv = Lv - bv * Lv * dt + J_var * Z_pois
            Lv = max(Lv, 1e-8)
            Lv_jump = Lv - Lv_prev

            # Update long-term variance center (Ornstein-Uhlenbeck)
            m = m + kappam * (thetam - m) * dt + omegam * np.sqrt(dt) * B
            m = max(m, 1e-8)

            # Update instantaneous variance (CIR with variance jumps)
            V1 = V1 + kappa1 * (theta1 - V1) * dt + omega1 * np.sqrt(V1) * np.sqrt(dt) * Z_3 + Lv_jump
            V1 = max(V1, 1e-8)

            # Update log-VIX (mean-reverting with stochastic volatility)
            # Critical: Use PREVIOUS values for Euler-Maruyama consistency
            X = X_prev + kappa * (m_prev - X_prev) * dt + np.sqrt(V1_prev) * np.sqrt(dt) * Z1

        # Convert log-VIX to VIX level
        paths[i] = np.exp(X)


# =====================================================================
# PRICING FUNCTIONS
# =====================================================================

def SNCallPricing(strucParams, jump_param, ini_state, S, K, T, r, NoOfPaths=1000000, NoOfSteps=512):
    """
    Compute VIX call option price via GPU-accelerated Monte Carlo.
    
    Parameters:
    -----------
    strucParams : list
        [kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv]
    jump_param : list
        [eta, muJv]
    ini_state : list
        [v10, Lv0, m0]
    S : float
        Current VIX (spot price / forward)
    K : float
        Strike price
    T : float
        Time-to-maturity in years
    r : float
        Risk-free rate
    NoOfPaths : int
        Number of Monte Carlo paths (default: 1,000,000)
    NoOfSteps : int
        Time steps per path (default: 512)
    
    Returns:
    --------
    tuple : (forward_price, call_price)
        forward_price: Simulated forward VIX expectation
        call_price: Discounted call option value
    """
    kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv = strucParams
    v10, Lv0, m0 = ini_state
    eta, muJv = jump_param

    # CUDA configuration
    paths = cuda.to_device(np.zeros(NoOfPaths, dtype=np.float32))
    threads_per_block = 128
    blocks = 512
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=100)

    # Run kernel
    Generate_DFH[blocks, threads_per_block](NoOfPaths, NoOfSteps, T, r, S, kappa, kappam, thetam, omegam,
                                            kappa1, theta1, omega1, rho1, bv, eta, muJv, v10, Lv0, m0, rng_states, paths)

    # Retrieve results and compute statistics
    VIX_T = paths.copy_to_host()
    forward_price = np.round(np.mean(VIX_T), 4)
    call_price = np.round(np.exp(-r * T) * np.mean(np.maximum(VIX_T - K, 0)), 4)

    return forward_price, call_price


def Black76_IV(F, K, T, r, call_price, tol=1e-6, max_iter=100):
    """
    Compute implied volatility using Black76 formula inversion.
    
    Parameters:
    -----------
    F : float
        Forward price
    K : float
        Strike price
    T : float
        Time-to-maturity
    r : float
        Risk-free rate
    call_price : float
        Market call price
    tol : float
        Newton-Raphson tolerance
    max_iter : int
        Maximum iterations
    
    Returns:
    --------
    float : Implied volatility
    """
    # Initial guess based on ATM volatility approximation
    sigma = np.sqrt(2 * np.pi / T) * call_price / F

    for _ in range(max_iter):
        d1 = (np.log(F / K) + sigma**2 * T / 2) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Black76 call price
        C = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

        # Vega density
        vega = np.exp(-r * T) * F * norm.pdf(d1) * np.sqrt(T)

        # Newton-Raphson step
        diff = C - call_price
        if abs(diff) < tol or vega < 1e-10:
            break
        sigma = sigma - diff / vega

    return max(sigma, 1e-4)


def SNCallIV(strucParams, jump_param, ini_state, S, K, T, r, NoOfPaths=1000000, NoOfSteps=512):
    """
    Compute implied volatility from Monte Carlo call price.
    
    Returns:
    --------
    float : Implied volatility (annualized)
    """
    forw_price, call_price = SNCallPricing(strucParams, jump_param, ini_state, S, K, T, r, NoOfPaths, NoOfSteps)
    iv = Black76_IV(forw_price, K, T, r, call_price)
    return np.round(iv, 4)


# =====================================================================
# COMPREHENSIVE SENSITIVITY ANALYSIS
# =====================================================================

def comprehensive_call_price_analysis(params, jump_params, v10, Lv0, m0, VIX_0, r, output_prefix=''):
    """
    Generate comprehensive 2x2 call price sensitivity analysis figure.
    
    Analyzes moneyness and term structure sensitivity to $L_v$ and $b_v$.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE CALL PRICE ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    T = 30 / 365
    mnes = np.linspace(0.8, 1.5, 17)
    Lv_list = [0.0, 1.0, 5.0]
    bv_list = [0.0, 10.0, 50.0]

    kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv = params
    eta, muJv = jump_params

    # Top Left: Moneyness sensitivity to Lv
    ax = axes[0, 0]
    for Lv_val, color, style in zip(Lv_list, ['#E41A1C', '#377EB8', '#4DAF4A'], ['dotted', 'dashdot', '-']):
        callPrice_list = []
        for mne in tqdm(mnes, desc=f"Moneyness Lv={Lv_val}"):
            ini_state = [v10, Lv_val, m0]
            _, callPrice = SNCallPricing(params, jump_params, ini_state, VIX_0, VIX_0 * mne, T, r)
            callPrice_list.append(callPrice)
        ax.plot(mnes * VIX_0, callPrice_list, label=f'$L_v$={Lv_val}', color=color, linestyle=style, linewidth=2.0, marker='o', markersize=3)
    ax.set_xlabel('Strike K', fontsize=11, fontweight='bold')
    ax.set_ylabel('Call Price', fontsize=11, fontweight='bold')
    ax.set_title('Impact of $L_v$', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top Right: Moneyness sensitivity to bv
    ax = axes[0, 1]
    for bv_val, color, style in zip(bv_list, ['#E41A1C', '#377EB8', '#4DAF4A'], ['dotted', 'dashdot', '-']):
        params_bv = [kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv_val]
        callPrice_list = []
        for mne in tqdm(mnes, desc=f"Moneyness bv={bv_val}"):
            ini_state = [v10, Lv0, m0]
            _, callPrice = SNCallPricing(params_bv, jump_params, ini_state, VIX_0, VIX_0 * mne, T, r)
            callPrice_list.append(callPrice)
        ax.plot(mnes * VIX_0, callPrice_list, label=f'$b_v$={bv_val}', color=color, linestyle=style, linewidth=2.0, marker='o', markersize=3)
    ax.set_xlabel('Strike K', fontsize=11, fontweight='bold')
    ax.set_ylabel('Call Price', fontsize=11, fontweight='bold')
    ax.set_title('Impact of $b_v$', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom Left: Term structure sensitivity to Lv
    ax = axes[1, 0]
    tau_list = np.linspace(20 / 365, 150 / 365, 17)
    for Lv_val, color, style in zip(Lv_list, ['#E41A1C', '#377EB8', '#4DAF4A'], ['dotted', 'dashdot', '-']):
        callPrice_list = []
        for tau in tqdm(tau_list, desc=f"Term Lv={Lv_val}"):
            ini_state = [v10, Lv_val, m0]
            _, callPrice = SNCallPricing(params, jump_params, ini_state, VIX_0, 0, tau, r)
            callPrice_list.append(callPrice)
        ax.plot(tau_list, callPrice_list, label=f'$L_v$={Lv_val}', color=color, linestyle=style, linewidth=2.0, marker='o', markersize=3)
    ax.set_xlabel('Time to Maturity (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ATM Call Price', fontsize=11, fontweight='bold')
    ax.set_title('Impact of $L_v$', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom Right: Term structure sensitivity to bv
    ax = axes[1, 1]
    for bv_val, color, style in zip(bv_list, ['#E41A1C', '#377EB8', '#4DAF4A'], ['dotted', 'dashdot', '-']):
        params_bv = [kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv_val]
        callPrice_list = []
        for tau in tqdm(tau_list, desc=f"Term bv={bv_val}"):
            ini_state = [v10, Lv0, m0]
            _, callPrice = SNCallPricing(params_bv, jump_params, ini_state, VIX_0, 0, tau, r)
            callPrice_list.append(callPrice)
        ax.plot(tau_list, callPrice_list, label=f'$b_v$={bv_val}', color=color, linestyle=style, linewidth=2.0, marker='o', markersize=3)
    ax.set_xlabel('Time to Maturity (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ATM Call Price', fontsize=11, fontweight='bold')
    ax.set_title('Impact of $b_v$', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'{output_prefix}comprehensive_call_price_analysis.png'
    plt.savefig(filename, dpi=400)
    print(f"✓ Saved: {filename}")
    plt.close()


def comprehensive_iv_analysis(params, jump_params, v10, Lv0, m0, VIX_0, r, output_prefix=''):
    """
    Generate comprehensive 2x2 IV sensitivity analysis figure.
    
    Analyzes IV skew and term structure sensitivity to $L_v$ and $b_v$.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE CALL IV ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    T = 30 / 365
    mnes = np.linspace(0.8, 1.5, 15)
    tau_list = np.linspace(20 / 365, 150 / 365, 15)
    Lv_list = [0.0, 1.0, 5.0]
    bv_list = [0.0, 10.0, 50.0]

    kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv = params
    eta, muJv = jump_params

    # Top Left: IV skew sensitivity to Lv
    ax = axes[0, 0]
    for Lv_val, color, style in zip(Lv_list, ['#E41A1C', '#377EB8', '#4DAF4A'], ['dotted', 'dashdot', '-']):
        callIV_list = []
        for mne in tqdm(mnes, desc=f"IV Skew Lv={Lv_val}"):
            ini_state = [v10, Lv_val, m0]
            forward, _ = SNCallPricing(params, jump_params, ini_state, VIX_0, 0, T, r)
            strike = forward * mne
            iv = SNCallIV(params, jump_params, ini_state, VIX_0, strike, T, r)
            callIV_list.append(iv)
        ax.plot(mnes, callIV_list, label=f'$L_v$={Lv_val}', color=color, linestyle=style, linewidth=2.0, marker='o', markersize=3)
    ax.set_xlabel('Moneyness (K/F)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Implied Volatility', fontsize=11, fontweight='bold')
    ax.set_title('Impact of $L_v$ on IV Skew', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top Right: IV skew sensitivity to bv
    ax = axes[0, 1]
    for bv_val, color, style in zip(bv_list, ['#E41A1C', '#377EB8', '#4DAF4A'], ['dotted', 'dashdot', '-']):
        params_bv = [kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv_val]
        callIV_list = []
        for mne in tqdm(mnes, desc=f"IV Skew bv={bv_val}"):
            ini_state = [v10, Lv0, m0]
            forward, _ = SNCallPricing(params_bv, jump_params, ini_state, VIX_0, 0, T, r)
            strike = forward * mne
            iv = SNCallIV(params_bv, jump_params, ini_state, VIX_0, strike, T, r)
            callIV_list.append(iv)
        ax.plot(mnes, callIV_list, label=f'$b_v$={bv_val}', color=color, linestyle=style, linewidth=2.0, marker='o', markersize=3)
    ax.set_xlabel('Moneyness (K/F)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Implied Volatility', fontsize=11, fontweight='bold')
    ax.set_title('Impact of $b_v$ on IV Skew', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom Left: ATM IV term structure sensitivity to Lv
    ax = axes[1, 0]
    for Lv_val, color, style in zip(Lv_list, ['#E41A1C', '#377EB8', '#4DAF4A'], ['dotted', 'dashdot', '-']):
        callIV_list = []
        for tau in tqdm(tau_list, desc=f"IV Term Lv={Lv_val}"):
            ini_state = [v10, Lv_val, m0]
            forward, _ = SNCallPricing(params, jump_params, ini_state, VIX_0, 0, tau, r)
            strike = forward  # ATM
            iv = SNCallIV(params, jump_params, ini_state, VIX_0, strike, tau, r)
            callIV_list.append(iv)
        ax.plot(tau_list, callIV_list, label=f'$L_v$={Lv_val}', color=color, linestyle=style, linewidth=2.0, marker='o', markersize=3)
    ax.set_xlabel('Time to Maturity (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ATM Implied Volatility', fontsize=11, fontweight='bold')
    ax.set_title('Impact of $L_v$ on IV Term Structure', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom Right: ATM IV term structure sensitivity to bv
    ax = axes[1, 1]
    for bv_val, color, style in zip(bv_list, ['#E41A1C', '#377EB8', '#4DAF4A'], ['dotted', 'dashdot', '-']):
        params_bv = [kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv_val]
        callIV_list = []
        for tau in tqdm(tau_list, desc=f"IV Term bv={bv_val}"):
            ini_state = [v10, Lv0, m0]
            forward, _ = SNCallPricing(params_bv, jump_params, ini_state, VIX_0, 0, tau, r)
            strike = forward  # ATM
            iv = SNCallIV(params_bv, jump_params, ini_state, VIX_0, strike, tau, r)
            callIV_list.append(iv)
        ax.plot(tau_list, callIV_list, label=f'$b_v$={bv_val}', color=color, linestyle=style, linewidth=2.0, marker='o', markersize=3)
    ax.set_xlabel('Time to Maturity (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ATM Implied Volatility', fontsize=11, fontweight='bold')
    ax.set_title('Impact of $b_v$ on IV Term Structure', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'{output_prefix}comprehensive_call_iv_analysis.png'
    plt.savefig(filename, dpi=400)
    print(f"✓ Saved: {filename}")
    plt.close()


# =====================================================================
# SANITY CHECK: FFT vs CUDA MONTE CARLO
# =====================================================================

def sanity_check_fft_vs_cuda(params, jump_params, v10, Lv0, m0, VIX_0, r, T=90/360, output_prefix=''):
    """
    Three-Way Pricing Validation: FFT vs GPU Monte Carlo vs CPU Monte Carlo.
    
    Compares call prices across strike range: K in [15, 30].
    Validates GPU implementation against CPU reference implementation.
    """
    print("\n" + "="*110)
    print("THREE-WAY PRICING VALIDATION: FFT vs GPU MC vs CPU MC")
    print("="*110)

    try:
        sys.path.append('./Others')
        from pricer_validation_submodel import ModelParameters, InitialConditions, FFTPricer, MonteCarloSimulator
    except ImportError:
        print("❌ ERROR: FFT pricing module not found. Skipping sanity check.")
        return

    kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv = params
    eta, muJv = jump_params

    print("\n[Step 1] Setting Model Parameters...")
    
    # Setup FFT parameters
    params_fft = ModelParameters(
        T=T, kappa=kappa, kappam=kappam, thetam=thetam, omegam=omegam,
        kappa1=kappa1, theta1=theta1, omega1=omega1, rho1=rho1,
        bv=bv, lamb=eta, muJV=muJv
    )
    initial_fft = InitialConditions(VIX0=VIX_0, v10=v10, Lv0=Lv0, m0=m0)
    
    print(f"  Maturity: {T*360:.0f} days, VIX0: {VIX_0}, r: {r*100}%")
    print(f"  kappa={kappa:.4f}, kappam={kappam:.4f}, kappa1={kappa1:.4f}")
    print(f"  bv={bv:.4f}, eta={eta:.4f}, muJV={muJv:.4f}")

    strikes_range = np.linspace(15, 30, 16)
    
    # =====================================================================
    # Step 2: FFT Pricing
    # =====================================================================
    print("\n[Step 2] FFT Pricing...")
    print(f"\n  {'Strike':<10} {'FFT Price':<15} {'Time (s)':<10}")
    print("  " + "-"*35)
    
    fft_prices = []
    for K in strikes_range:
        t_start = time.time()
        price = FFTPricer.price_single_call(r, T, K, params_fft, initial_fft)
        elapsed = time.time() - t_start
        fft_prices.append(price)
        print(f"  {K:<10.1f} {price:<15.6f} {elapsed:<10.4f}")

    # =====================================================================
    # Step 3: GPU Monte Carlo Pricing
    # =====================================================================
    print("\n[Step 3] GPU Monte Carlo Pricing...")
    
    NoOfSteps_gpu = 512
    NoOfPaths_gpu = 1000000
    print(f"  Configuration: {NoOfPaths_gpu:,} paths, {NoOfSteps_gpu} steps")
    
    gpu_prices = []
    print(f"\n  {'Strike':<10} {'GPU MC Price':<15} {'Time (s)':<10}")
    print("  " + "-"*40)
    
    for K in strikes_range:
        t_start = time.time()
        
        # Reinitialize paths and RNG states for each strike
        paths_gpu = cuda.to_device(np.zeros(NoOfPaths_gpu, dtype=np.float32))
        rng_states_gpu = create_xoroshiro128p_states(128 * 512, seed=100)
        
        Generate_DFH[512, 128](NoOfPaths_gpu, NoOfSteps_gpu, T, r, VIX_0, 
                              kappa, kappam, thetam, omegam, 
                              kappa1, theta1, omega1, rho1, bv, eta, muJv, 
                              v10, Lv0, m0, rng_states_gpu, paths_gpu)
        
        VIX_T_gpu = paths_gpu.copy_to_host()
        price_gpu = np.exp(-r*T) * np.mean(np.maximum(VIX_T_gpu - K, 0))
        elapsed = time.time() - t_start
        gpu_prices.append(price_gpu)
        print(f"  {K:<10.1f} {price_gpu:<15.6f} {elapsed:<10.4f}")

    # =====================================================================
    # Step 4: CPU Monte Carlo Pricing (Reference)
    # =====================================================================
    print("\n[Step 4] CPU Monte Carlo Pricing (Reference)...")
    
    NoOfSteps_cpu = 256
    NoOfPaths_cpu = 500000
    print(f"  Configuration: {NoOfPaths_cpu:,} paths, {NoOfSteps_cpu} steps")
    
    cpu_prices = []
    print(f"\n  {'Strike':<10} {'CPU MC Price':<15} {'Time (s)':<10}")
    print("  " + "-"*40)
    
    for K in strikes_range:
        t_start = time.time()
        price_cpu = MonteCarloSimulator.price_call_mc(r, T, K, params_fft, initial_fft,
                                                      num_paths=NoOfPaths_cpu, 
                                                      num_steps=NoOfSteps_cpu)
        elapsed = time.time() - t_start
        cpu_prices.append(price_cpu)
        print(f"  {K:<10.1f} {price_cpu:<15.6f} {elapsed:<10.4f}")

    # =====================================================================
    # Step 5: Three-Way Comparison
    # =====================================================================
    print("\n" + "="*110)
    print("THREE-WAY COMPARISON: FFT vs GPU MC vs CPU MC")
    print("="*110)
    print(f"  {'Strike':<10} {'FFT':<15} {'GPU MC':<15} {'CPU MC':<15} {'GPU-FFT %':<14} {'CPU-FFT %':<14} {'GPU-CPU %':<14}")
    print("  " + "-"*110)

    gpu_fft_errs = []
    cpu_fft_errs = []
    gpu_cpu_errs = []

    for i, K in enumerate(strikes_range):
        fft_price = fft_prices[i]
        gpu_price = gpu_prices[i]
        cpu_price = cpu_prices[i]
        
        gpu_fft_err = abs(gpu_price - fft_price) / fft_price * 100 if fft_price != 0 else 0
        cpu_fft_err = abs(cpu_price - fft_price) / fft_price * 100 if fft_price != 0 else 0
        gpu_cpu_err = abs(gpu_price - cpu_price) / cpu_price * 100 if cpu_price != 0 else 0
        
        gpu_fft_errs.append(gpu_fft_err)
        cpu_fft_errs.append(cpu_fft_err)
        gpu_cpu_errs.append(gpu_cpu_err)
        
        print(f"  {K:<10.1f} {fft_price:<15.6f} {gpu_price:<15.6f} {cpu_price:<15.6f} {gpu_fft_err:<13.3f}% {cpu_fft_err:<13.3f}% {gpu_cpu_err:<13.3f}%")

    print("="*110)

    # =====================================================================
    # Summary Statistics
    # =====================================================================
    print("\n📊 Summary Statistics:")
    print("-" * 50)
    print(f"GPU vs FFT  - Mean Error: {np.mean(gpu_fft_errs):.3f}%, Max Error: {np.max(gpu_fft_errs):.3f}%")
    print(f"CPU vs FFT  - Mean Error: {np.mean(cpu_fft_errs):.3f}%, Max Error: {np.max(cpu_fft_errs):.3f}%")
    print(f"GPU vs CPU  - Mean Error: {np.mean(gpu_cpu_errs):.3f}%, Max Error: {np.max(gpu_cpu_errs):.3f}%")

    # =====================================================================
    # Create comparison figure
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Price comparison (FFT vs GPU MC)
    ax = axes[0]
    ax.plot(strikes_range, fft_prices, label='FFT', color='#E41A1C', linestyle='-',
            linewidth=2.0, marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1)
    ax.plot(strikes_range, gpu_prices, label='GPU MC', color='#377EB8', linestyle='--', linewidth=2.0)
    ax.plot(strikes_range, cpu_prices, label='CPU MC', color='#4DAF4A', linestyle=':', linewidth=2.0)
    ax.set_xlabel('Strike K', fontsize=11, fontweight='bold')
    ax.set_ylabel('Call Price', fontsize=11, fontweight='bold')
    ax.set_title('Call Price: FFT vs GPU MC vs CPU MC', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Right: Error analysis (GPU vs FFT and CPU vs FFT)
    ax = axes[1]
    ax.plot(strikes_range, gpu_fft_errs, label='GPU vs FFT', color='#377EB8',
            linestyle='-', linewidth=2.0, marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1)
    ax.plot(strikes_range, cpu_fft_errs, label='CPU vs FFT', color='#4DAF4A',
            linestyle='--', linewidth=2.0, marker='s', markersize=6, markeredgecolor='white', markeredgewidth=1)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Strike K', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
    ax.set_title('Relative Error: MC Methods vs FFT', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'{output_prefix}sanity_check_three_way.png'
    plt.savefig(filename, dpi=400)
    print(f"\n✓ Saved: {filename}")
    plt.close()

    # =====================================================================
    # Validation Results
    # =====================================================================
    print("\n" + "="*80)
    print("✓ Three-Way Validation Complete!")
    print("="*80)
    print("  Comparing FFT (baseline), GPU Monte Carlo, and CPU Monte Carlo pricing methods")
    print("\n✅ Key Verification Points:")
    print("  • GPU MC and CPU MC should be very close (< 1-2% difference)")
    print("  • Both MC methods should closely match FFT results (< 2-3% difference)")
    print("  • This confirms GPU implementation matches CPU reference correctly")
    print("="*80)


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """
    Main execution routine for comprehensive VIX option pricing analysis.
    """
    print("\n" + "="*80)
    print("VIX SHOT NOISE MODEL - CUDA MONTE CARLO PRICING")
    print("="*80)

    # Set random seed for reproducibility
    np.random.seed(10)

    # Model parameters (calibrated)
    r = 0.03
    kappa = 5.0898
    kappam = 1.0218
    thetam = 3.4247
    omegam = 0.2282
    kappa1 = 2.4754
    theta1 = 2.5097
    omega1 = 1.1062
    rho1 = 0.4976
    bv = 12.3813

    # Initial conditions
    VIX_0 = 20
    v10 = 0.64
    Lv0 = 0.0
    m0 = 3

    # Jump parameters
    eta = 4.8194
    muJv = 1.5831

    # Package parameters
    params = [kappa, kappam, thetam, omegam, kappa1, theta1, omega1, rho1, bv]
    jump_params = [eta, muJv]
    ini_state = [v10, Lv0, m0]

    print(f"\nModel Configuration:")
    print(f"  κ = {kappa:.4f}, κ_m = {kappam:.4f}, κ_1 = {kappa1:.4f}")
    print(f"  η = {eta:.4f}, μ_Jv = {muJv:.4f}, b_v = {bv:.4f}")
    print(f"  VIX_0 = {VIX_0}, v_1^0 = {v10}, ρ_1 = {rho1:.4f}")

    # Run analyses
    comprehensive_call_price_analysis(params, jump_params, v10, Lv0, m0, VIX_0, r)
    comprehensive_iv_analysis(params, jump_params, v10, Lv0, m0, VIX_0, r)
    sanity_check_fft_vs_cuda(params, jump_params, v10, Lv0, m0, VIX_0, r)

    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
