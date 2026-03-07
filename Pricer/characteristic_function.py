"""
Characteristic Function Solver
===============================

Computes the characteristic function for the reduced VIX shot noise model
by solving a system of coupled Riccati-type ODEs for affine coefficients.

Author: VIX Spike VolOfVol Research Team
Date: February 2026
"""

import numpy as np
import scipy.interpolate as interpolate
from typing import Callable
from model_config import ModelParameters, InitialConditions
from toolkit import odeRK4


class CharacteristicFunctionSolver:
    """
    Solves the characteristic function for the reduced shot noise VIX model.
    
    The characteristic function is decomposed into affine coefficients
    A(τ,u), B(τ,u), D(τ,u), E(τ,u), F(τ,u) which solve a system
    of coupled Riccati-type ODEs.
    
    Note: Coefficient C is excluded as there is no level shot noise.
    """
    
    @staticmethod
    def coef_A(tau: float, u: np.ndarray, kappa: float) -> np.ndarray:
        """
        Compute coefficient A(τ,u) for the log-VIX initial value.
        
        This represents the sensitivity to the initial log-VIX level,
        decaying exponentially due to mean-reversion.
        
        Parameters:
        -----------
        tau : float
            Time horizon
        u : array_like
            Frequency domain variable
        kappa : float
            Mean-reversion speed
            
        Returns:
        --------
        A : ndarray
            Coefficient values for each u
        """
        return 1j * u * np.exp(-kappa * tau)
    
    @staticmethod
    def coef_E(u: np.ndarray, T: float, kappa: float, 
               kappam: float, omegam: float) -> Callable:
        """
        Solve ODE for coefficient E(τ,u) corresponding to long-term center m.
        
        The ODE incorporates feedback from the mean-reverting VIX process
        and the OU dynamics of the variance center.
        
        Parameters:
        -----------
        u : array_like
            Frequency domain variable
        T : float
            Terminal time
        kappa, kappam, omegam : float
            Model parameters
            
        Returns:
        --------
        E_func : callable
            Interpolated function E(t,u) for t ∈ [0,T]
        """
        E_0_u = 0
        nU = len(u)
        
        def RHS_E(t, y):
            """Right-hand side of the ODE for E(t,u)."""
            tmp1 = kappa * CharacteristicFunctionSolver.coef_A(t, u, kappa)
            tmp2 = -kappam * y
            # For OU process: no quadratic term
            tmp3 = 0
            return tmp1 + tmp2 + tmp3
        
        time, E_tau_u = odeRK4(RHS_E, 0, T, E_0_u, 100, nU)
        E_tau_u_func = interpolate.interp1d(time, E_tau_u, kind='cubic')
        
        return E_tau_u_func
    
    @staticmethod
    def coef_B(u: np.ndarray, T: float, kappa: float,
               kappai: float, omegai: float, rhoi: float) -> Callable:
        """
        Solve ODE for coefficient B(τ,u) for instantaneous variance v1.
        
        This coefficient captures the stochastic volatility feedback,
        including correlation effects and vol-of-vol dynamics.
        
        Parameters:
        -----------
        u : array_like
            Frequency domain variable
        T : float
            Terminal time
        kappai, omegai, rhoi : float
            Parameters for variance process (mean-reversion, vol-of-vol, correlation)
            
        Returns:
        --------
        B_func : callable
            Interpolated function B(t,u) for t ∈ [0,T]
        """
        B_0_u = 0
        nU = len(u)
        
        def RHS_B(t, y):
            """Right-hand side of the Riccati ODE for B(t,u)."""
            A_t = CharacteristicFunctionSolver.coef_A(t, u, kappa)
            tmp1 = -kappai * y  # Mean-reversion term
            tmp2 = 0.5 * A_t**2  # Diffusion contribution from log-VIX
            tmp3 = 0.5 * omegai**2 * y**2  # CIR-type vol-of-vol term
            tmp4 = rhoi * omegai * A_t * y  # Correlation term
            return tmp1 + tmp2 + tmp3 + tmp4
        
        time, B_tau_u = odeRK4(RHS_B, 0, T, B_0_u, 100, nU)
        B_tau_u_func = interpolate.interp1d(time, B_tau_u, kind='cubic')
        
        return B_tau_u_func
    
    @staticmethod
    def coef_D(u: np.ndarray, T: float, kappa: float, kappa1: float,
               omega1: float, rho1: float, bv: float) -> Callable:
        """
        Solve ODE for coefficient D(τ,u) for shot noise component L^v.
        
        This captures the effect of variance jumps on the option price.
        
        Parameters:
        -----------
        u : array_like
            Frequency domain variable
        T : float
            Terminal time
        kappa, kappa1, omega1, rho1 : float
            Model parameters
        bv : float
            Shot noise decay rate for variance jumps
            
        Returns:
        --------
        D_func : callable
            Interpolated function D(t,u) for t ∈ [0,T]
        """
        D_0_u = 0
        nU = len(u)
        
        # Precompute B coefficient
        coef_B_func = CharacteristicFunctionSolver.coef_B(
            u, T, kappa, kappa1, omega1, rho1
        )
        
        def RHS_D(t, y):
            """Right-hand side of ODE for D(t,u)."""
            tmp1 = -bv * coef_B_func(t)  # Coupling with variance process
            tmp2 = -bv * y               # Self-decay of shot noise
            return tmp1 + tmp2
        
        time, D_tau_u = odeRK4(RHS_D, 0, T, D_0_u, 100, nU)
        D_tau_u_func = interpolate.interp1d(time, D_tau_u, kind='cubic')
        
        return D_tau_u_func
    
    @staticmethod
    def coef_F(u: np.ndarray, params: ModelParameters) -> Callable:
        """
        Solve ODE for coefficient F(τ,u) - the constant term.
        
        This aggregates contributions from mean-reversion targets and
        jump compensation terms for variance jumps only.
        
        Parameters:
        -----------
        u : array_like
            Frequency domain variable
        params : ModelParameters
            Complete set of model parameters
            
        Returns:
        --------
        F_func : callable
            Interpolated function F(t,u) for t ∈ [0,T]
        """
        F_0_u = 0
        nU = len(u)
        
        # Unpack parameters
        T = params.T
        kappa = params.kappa
        kappam, thetam, omegam = params.kappam, params.thetam, params.omegam
        kappa1, theta1, omega1, rho1 = params.kappa1, params.theta1, params.omega1, params.rho1
        bv = params.bv
        lamb, muJV = params.lamb, params.muJV
        
        # Precompute all necessary coefficient functions
        coef_B_func = CharacteristicFunctionSolver.coef_B(u, T, kappa, kappa1, omega1, rho1)
        coef_D_func = CharacteristicFunctionSolver.coef_D(u, T, kappa, kappa1, omega1, rho1, bv)
        coef_E_func = CharacteristicFunctionSolver.coef_E(u, T, kappa, kappam, omegam)
        
        def RHS_F(t, y):
            """
            Right-hand side of ODE for F(t,u).
            
            Includes:
            - Mean-reversion contribution from variance
            - Mean-reversion contribution from long-term center
            - Jump compensation from variance jumps only (no level jumps)
            """
            # Variance mean-reversion contribution
            tmp1 = kappa1 * theta1 * coef_B_func(t)
            
            # Long-term center contributions (OU process)
            tmp2 = kappam * thetam * coef_E_func(t)
            tmp2 += 0.5 * omegam**2 * coef_E_func(t)**2  # OU adjustment
            
            # Jump compensation terms (variance jumps only)
            # For variance jumps: exponential distribution with mean muJV
            jump_v = 1.0 / (1 - (coef_B_func(t) + coef_D_func(t)) * muJV)
            
            # Only variance jump contribution (no level jumps)
            tmp3 = lamb * (jump_v - 1)
            
            return tmp1 + tmp2 + tmp3
        
        time, F_tau_u = odeRK4(RHS_F, 0, T, F_0_u, 100, nU)
        F_tau_u_func = interpolate.interp1d(time, F_tau_u, kind='cubic')
        
        return F_tau_u_func
    
    @classmethod
    def compute_characteristic_function(cls, u: np.ndarray, 
                                       params: ModelParameters,
                                       initial: InitialConditions) -> np.ndarray:
        """
        Compute the full characteristic function of log-VIX at maturity.
        
        The characteristic function has the affine form:
        φ(u) = exp(A·vix0 + B·v10 + D·Lv0 + E·m0 + F)
        
        Note: No C term as there is no level shot noise.
        
        Parameters:
        -----------
        u : array_like
            Frequency domain variable
        params : ModelParameters
            Model parameter configuration
        initial : InitialConditions
            Initial state of all processes
            
        Returns:
        --------
        chf : ndarray
            Characteristic function values for each u
        """
        T = params.T
        kappa = params.kappa
        kappam, omegam = params.kappam, params.omegam
        kappa1, theta1, omega1, rho1 = params.kappa1, params.theta1, params.omega1, params.rho1
        bv = params.bv
        
        # Convert initial VIX to log scale
        vix0 = np.log(initial.VIX0)
        
        # Compute all coefficient functions
        coef_B_func = cls.coef_B(u, T, kappa, kappa1, omega1, rho1)
        coef_D_func = cls.coef_D(u, T, kappa, kappa1, omega1, rho1, bv)
        coef_E_func = cls.coef_E(u, T, kappa, kappam, omegam)
        coef_F_func = cls.coef_F(u, params)
        
        # Evaluate at terminal time T (no C term)
        exponent = (
            cls.coef_A(T, u, kappa) * vix0 +
            coef_B_func(T) * initial.v10 +
            coef_D_func(T) * initial.Lv0 +
            coef_E_func(T) * initial.m0 +
            coef_F_func(T)
        )
        
        return np.exp(exponent)
