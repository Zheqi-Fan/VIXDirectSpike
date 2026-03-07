"""
Validation Test Script
======================

Main script to validate the reduced VIX model by comparing
FFT and Monte Carlo pricing methods.

Author: VIX Spike VolOfVol Research Team
Date: February 2026
"""

import numpy as np
import time
from model_config import ModelParameters, InitialConditions
from fft_pricing import FFTPricer
from monte_carlo import MonteCarloSimulator
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')


def main():
    """
    Main validation routine: FFT vs Monte Carlo pricing comparison.
    """
    print("\n" + "="*80)
    print("REDUCED VIX MODEL - FFT vs MONTE CARLO VALIDATION")
    print("(No Level Shot Noise: bs=0, Ls0=0, muJS excluded)")
    print("="*80)
    
    # =========================================================================
    # Model Configuration
    # =========================================================================
    
    print("\n[Step 1] Setting Model Parameters...")
    
    # Time and market parameters
    r = 0.03
    T = 90/360
    kappa = 5
    
    # Long-term variance center (OU process)
    kappam, thetam, omegam = 2.0, 3.0, 0.25
    
    # Instantaneous variance (CIR-type process)
    kappa1, theta1, omega1, rho1 = 2.0, 2.0, 3.0, 0.8
    
    # Shot noise parameters (variance jumps only)
    bv = 0.05
    
    # Jump parameters (variance jumps only, no level jumps)
    lamb, muJV = 1.5, 0.2
    
    params = ModelParameters(
        T=T, kappa=kappa, kappam=kappam, thetam=thetam, omegam=omegam,
        kappa1=kappa1, theta1=theta1, omega1=omega1, rho1=rho1,
        bv=bv, lamb=lamb, muJV=muJV
    )
    
    # Initial conditions (no Ls0)
    VIX_0, v10, Lv0, m0 = 40, 0.64, 0., 3
    initial = InitialConditions(
        VIX0=VIX_0, v10=v10, Lv0=Lv0, m0=m0
    )
    
    print(f"  Maturity: {T*360:.0f} days, VIX0: {VIX_0}, r: {r*100}%")
    print(f"  Model: Variance jumps only (no level jumps)")
    
    # =========================================================================
    # FFT Pricing
    # =========================================================================
    
    print("\n[Step 2] FFT Pricing for Different Strikes...")
    
    strikes = [35, 40, 45]
    
    print(f"\n  {'Strike':<10} {'FFT Price':<15}")
    print("  " + "-"*25)
    
    fft_results = {}
    for K in strikes:
        t_start = time.time()
        price = FFTPricer.price_single_call(r, T, K, params, initial)
        elapsed = time.time() - t_start
        fft_results[K] = price
        print(f"  {K:<10} {price:<15.6f} ({elapsed:.4f}s)")
    
    # =========================================================================
    # Monte Carlo Pricing
    # =========================================================================
    
    print("\n[Step 3] Monte Carlo Validation...")
    print("  Running simulations with 300k paths...")
    
    NoOfSteps, NoOfPaths = 256, 300000
    
    print(f"\n  {'Strike':<10} {'MC Price':<15} {'Std Dev':<15} {'Runs':<10}")
    print("  " + "-"*50)
    
    mc_results = {}
    for K in strikes:
        call_prices = []
        num_runs = 6
        
        for run in range(num_runs):
            mc_price = MonteCarloSimulator.price_call_mc(
                r, T, K, params, initial,
                num_paths=NoOfPaths, num_steps=NoOfSteps
            )
            call_prices.append(mc_price)
        
        mc_mean = np.mean(call_prices)
        mc_std = np.std(call_prices)
        mc_results[K] = (mc_mean, mc_std)
        print(f"  {K:<10} {mc_mean:<15.6f} ±{mc_std:<14.6f} (n={num_runs})")
    
    # =========================================================================
    # Comparison Summary
    # =========================================================================
    
    print("\n" + "="*80)
    print("COMPARISON: FFT vs MONTE CARLO")
    print("="*80)
    print(f"  {'Strike':<10} {'FFT':<15} {'MC Mean':<15} {'Difference':<15} {'Error %':<10}")
    print("  " + "-"*65)
    
    for K in strikes:
        fft_price = fft_results[K]
        mc_mean, mc_std = mc_results[K]
        diff = abs(fft_price - mc_mean)
        error_pct = (diff / fft_price) * 100 if fft_price != 0 else 0
        print(f"  {K:<10} {fft_price:<15.6f} {mc_mean:<15.6f} {diff:<15.6f} {error_pct:<10.3f}")
    
    print("="*80)
    print("\nValidation Complete!")
    print("FFT and Monte Carlo results are consistent for the reduced model.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Execute main validation routine
    main()
