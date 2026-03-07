"""
Model Validation Framework
===========================

Validates pricing accuracy by comparing FFT and Monte Carlo methods.

Author: VIX Spike VolOfVol Research Team
Date: February 2026
"""

import numpy as np
import pandas as pd
import time
from typing import List
from tqdm import tqdm
from model_config import ModelParameters, InitialConditions
from fft_pricing import FFTPricer
from monte_carlo import MonteCarloSimulator


class ModelValidator:
    """
    Comprehensive validation suite for the reduced pricing model.
    
    Compares FFT and Monte Carlo results across multiple strikes
    to verify implementation correctness.
    """
    
    @staticmethod
    def validate_pricing(strikes: List[float],
                        r: float, T: float,
                        params: ModelParameters,
                        initial: InitialConditions,
                        num_mc_runs: int = 5,
                        verbose: bool = True) -> pd.DataFrame:
        """
        Validate pricing accuracy by comparing FFT and MC methods.
        
        Parameters:
        -----------
        strikes : list of float
            Strike prices to test
        r : float
            Risk-free rate
        T : float
            Time to maturity
        params : ModelParameters
            Model configuration
        initial : InitialConditions
            Initial state
        num_mc_runs : int, optional
            Number of MC replications for variance reduction (default: 5)
        verbose : bool, optional
            Print progress information (default: True)
            
        Returns:
        --------
        results : DataFrame
            Comparison table with columns:
            - Strike
            - FFT_Price
            - MC_Price
            - MC_Std (standard deviation across runs)
            - Diff (absolute difference)
            - Rel_Error (percentage error)
        """
        results = []
        
        for K in strikes:
            if verbose:
                print(f"\n{'='*70}")
                print(f"Validating Strike K = {K}")
                print(f"{'='*70}")
            
            # FFT pricing (deterministic)
            t_start = time.time()
            fft_price = FFTPricer.price_single_call(r, T, K, params, initial)
            fft_time = time.time() - t_start
            
            if verbose:
                print(f"FFT Price: {fft_price:.6f} (computed in {fft_time:.4f}s)")
            
            # Monte Carlo pricing (multiple runs for robustness)
            mc_prices = []
            if verbose:
                print(f"\nRunning {num_mc_runs} Monte Carlo replications...")
            
            for run in tqdm(range(num_mc_runs), disable=not verbose):
                t_start = time.time()
                mc_price = MonteCarloSimulator.price_call_mc(
                    r, T, K, params, initial
                )
                mc_prices.append(mc_price)
                mc_time = time.time() - t_start
                if verbose:
                    print(f"  Run {run+1}: {mc_price:.6f} (time: {mc_time:.2f}s)")
            
            mc_mean = np.mean(mc_prices)
            mc_std = np.std(mc_prices)
            diff = abs(fft_price - mc_mean)
            rel_error = (diff / fft_price) * 100 if fft_price != 0 else np.nan
            
            if verbose:
                print(f"\nMC Average: {mc_mean:.6f} ± {mc_std:.6f}")
                print(f"Difference: {diff:.6f} ({rel_error:.2f}%)")
            
            results.append({
                'Strike': K,
                'FFT_Price': fft_price,
                'MC_Price': mc_mean,
                'MC_Std': mc_std,
                'Diff': diff,
                'Rel_Error (%)': rel_error
            })
        
        return pd.DataFrame(results)
