"""
FFT-Based Option Pricing
=========================

Implements the Carr-Madan FFT method for European VIX option pricing.

Author: VIX Spike VolOfVol Research Team
Date: February 2026
"""

import numpy as np
from typing import Tuple
from model_config import ModelParameters, InitialConditions
from characteristic_function import CharacteristicFunctionSolver


class FFTPricer:
    """
    Implements the Carr-Madan FFT method for European option pricing.
    
    This approach transforms the pricing problem to Fourier space,
    leveraging the Fast Fourier Transform for computational efficiency.
    """
    
    @staticmethod
    def price_call_fft(r: float, T: float, params: ModelParameters,
                       initial: InitialConditions, K: float,
                       alpha: float = 1.5, eta: float = 0.25,
                       n: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Price European call option using FFT method.
        
        Parameters:
        -----------
        r : float
            Risk-free interest rate
        T : float
            Time to maturity
        params : ModelParameters
            Model configuration
        initial : InitialConditions
            Initial state variables
        K : float
            Strike price
        alpha : float, optional
            Damping parameter for Fourier inversion (default: 1.5)
        eta : float, optional
            Step size in frequency domain (default: 0.25)
        n : int, optional
            Power of 2 for FFT size: N = 2^n (default: 8)
            
        Returns:
        --------
        km : ndarray
            Array of log-strikes
        call_prices : ndarray
            Corresponding call option prices
            
        Notes:
        ------
        The damping parameter α must satisfy α > 0 for call options.
        Larger n increases accuracy but also computational cost.
        """
        N = 2**n
        
        # Step-size in log-strike space (from Nyquist relation)
        lda = (2 * np.pi / N) / eta
        
        # Center the grid around desired strike K
        beta = np.log(K)
        
        # Construct frequency grid
        nu_j = np.arange(N) * eta
        
        # Discount factor
        df = np.exp(-r * T)
        
        # Compute modified characteristic function in frequency domain
        chf_values = CharacteristicFunctionSolver.compute_characteristic_function(
            nu_j - (alpha + 1) * 1j, params, initial
        )
        
        # Apply damping and normalization
        psi_nu = chf_values / ((alpha + 1j * nu_j) * (alpha + 1 + 1j * nu_j))
        
        # Construct log-strike grid
        km = beta + lda * np.arange(N)
        
        # Simpson's rule weights for trapezoidal integration
        w = eta * np.ones(N)
        w[0] = eta / 2
        
        # Assemble integrand with phase factor
        x_input = np.exp(-1j * beta * nu_j) * df * psi_nu * w
        
        # Apply FFT
        y_output = np.fft.fft(x_input)
        
        # Extract real part and apply final multiplier
        multiplier = np.exp(-alpha * km) / np.pi
        call_prices = multiplier * np.real(y_output)
        
        return km, call_prices
    
    @staticmethod
    def price_single_call(r: float, T: float, K: float,
                         params: ModelParameters,
                         initial: InitialConditions) -> float:
        """
        Convenience function to price a single call option.
        
        Parameters:
        -----------
        r : float
            Risk-free rate
        T : float
            Time to maturity
        K : float
            Strike price
        params : ModelParameters
            Model parameters
        initial : InitialConditions
            Initial conditions
            
        Returns:
        --------
        price : float
            Call option price (rounded to 6 decimals)
        """
        eta, alpha, n = 0.25, 1.5, 8
        km, cT_km = FFTPricer.price_call_fft(r, T, params, initial, K, alpha, eta, n)
        return np.round(cT_km[0], 6)
