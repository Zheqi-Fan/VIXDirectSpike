"""
Monte Carlo Simulation Module
==============================

Implements Monte Carlo path simulation for the reduced VIX shot noise model
using Euler-Maruyama discretization.

Author: VIX Spike VolOfVol Research Team
Date: February 2026
"""

import numpy as np
from typing import Tuple
from model_config import ModelParameters, InitialConditions


class MonteCarloSimulator:
    """
    Monte Carlo path simulator for the reduced shot noise VIX model.
    
    Uses Euler-Maruyama discretization with truncation to ensure
    positivity of variance processes.
    """
    
    @staticmethod
    def generate_paths(num_paths: int, num_steps: int,
                      params: ModelParameters,
                      initial: InitialConditions) -> Tuple[np.ndarray, ...]:
        """
        Generate sample paths for all state variables in the reduced model.
        
        Parameters:
        -----------
        num_paths : int
            Number of Monte Carlo paths
        num_steps : int
            Number of time steps
        params : ModelParameters
            Model configuration
        initial : InitialConditions
            Initial state
            
        Returns:
        --------
        time : ndarray
            Time grid
        VIX : ndarray (num_paths × num_steps+1)
            VIX paths (not log-transformed)
        Lv : ndarray
            Shot noise paths for variance jumps
            
        Notes:
        ------
        - Applies reflection at zero boundary for variance processes
        - Only variance jumps are simulated (no level jumps)
        """
        T = params.T
        kappa = params.kappa
        kappam, thetam, omegam = params.kappam, params.thetam, params.omegam
        kappa1, theta1, omega1, rho1 = params.kappa1, params.theta1, params.omega1, params.rho1
        bv = params.bv
        lamb, muJV = params.lamb, params.muJV
        
        dt = T / float(num_steps)
        time = np.zeros(num_steps + 1)
        
        # =====================================================================
        # Generate all random innovations upfront
        # =====================================================================
        
        # Brownian motion innovations
        B = np.random.normal(0.0, 1.0, [num_paths, num_steps])   # For m process
        Z1 = np.random.normal(0.0, 1.0, [num_paths, num_steps])  # For log-VIX
        Z3 = np.random.normal(0.0, 1.0, [num_paths, num_steps])  # Independent for v1
        
        # Construct correlated Brownian motion for variance
        Z_3 = rho1 * Z1 + np.sqrt(1.0 - rho1**2) * Z3
        
        # Jump components: Poisson arrivals and exponential jump sizes (variance only)
        Z_Pois = np.random.poisson(lamb * dt, [num_paths, num_steps])
        J_var = np.random.exponential(muJV, [num_paths, num_steps])
        
        # =====================================================================
        # Initialize state variable arrays
        # =====================================================================
        
        v1 = np.zeros([num_paths, num_steps + 1])
        m = np.zeros([num_paths, num_steps + 1])
        X = np.zeros([num_paths, num_steps + 1])  # log-VIX
        Lv = np.zeros([num_paths, num_steps + 1])
        
        # Set initial conditions
        v1[:, 0] = initial.v10
        m[:, 0] = initial.m0
        X[:, 0] = np.log(initial.VIX0)
        Lv[:, 0] = initial.Lv0
        
        # =====================================================================
        # Euler-Maruyama time-stepping
        # =====================================================================
        
        for i in range(num_steps):
            # -----------------------------------------------------------------
            # Update shot noise for variance (L^v)
            # -----------------------------------------------------------------
            Lv[:, i+1] = (Lv[:, i] - bv * Lv[:, i] * dt + 
                         J_var[:, i] * Z_Pois[:, i])
            
            # -----------------------------------------------------------------
            # Update long-term variance center (m) - OU process
            # -----------------------------------------------------------------
            m[:, i+1] = (m[:, i] + 
                        kappam * (thetam - m[:, i]) * dt + 
                        omegam * np.sqrt(dt) * B[:, i])
            m[:, i+1] = np.maximum(m[:, i+1], 1e-6)  # Truncation at boundary
            
            # -----------------------------------------------------------------
            # Update instantaneous variance (v1) - CIR-type with jumps
            # -----------------------------------------------------------------
            v1[:, i+1] = (v1[:, i] + 
                         kappa1 * (theta1 - v1[:, i]) * dt +
                         omega1 * np.sqrt(v1[:, i]) * np.sqrt(dt) * Z_3[:, i] +
                         (Lv[:, i+1] - Lv[:, i]))
            v1[:, i+1] = np.maximum(v1[:, i+1], 1e-6)  # Truncation at boundary
            
            # -----------------------------------------------------------------
            # Update log-VIX (X) - Mean-reverting with stochastic vol (no level jumps)
            # -----------------------------------------------------------------
            X[:, i+1] = (X[:, i] + 
                        kappa * (m[:, i] - X[:, i]) * dt +
                        np.sqrt(v1[:, i]) * np.sqrt(dt) * Z1[:, i])
            
            time[i+1] = time[i] + dt
        
        # Transform to VIX level (exponential of log-VIX)
        VIX = np.exp(X)
        
        return time, VIX, Lv
    
    @staticmethod
    def price_call_mc(r: float, T: float, K: float,
                     params: ModelParameters,
                     initial: InitialConditions,
                     num_paths: int = int(1e6),
                     num_steps: int = 256) -> float:
        """
        Price European call option via Monte Carlo simulation.
        
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
            Initial state
        num_paths : int, optional
            Number of simulation paths (default: 1,000,000)
        num_steps : int, optional
            Number of time steps (default: 256)
            
        Returns:
        --------
        call_price : float
            Monte Carlo estimate of call option price
            
        Notes:
        ------
        Standard error decreases as 1/√num_paths.
        """
        time, VIX, Lv = MonteCarloSimulator.generate_paths(
            num_paths, num_steps, params, initial
        )
        
        # Compute discounted payoff
        payoff = np.maximum(0, VIX[:, -1] - K)
        call_price = np.exp(-r * T) * np.mean(payoff)
        
        return call_price
