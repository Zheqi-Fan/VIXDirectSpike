"""
Model Configuration Module
==========================

Defines data structures for model parameters and initial conditions
for the reduced VIX shot noise model (variance jumps only).

Author: VIX Spike VolOfVol Research Team
Date: February 2026
"""

from typing import List
from dataclasses import dataclass


@dataclass
class ModelParameters:
    """
    Encapsulates all model parameters for the reduced VIX dynamics.
    
    Attributes:
    -----------
    T : float
        Time to maturity (in years)
    kappa : float
        Mean-reversion speed of log-VIX process
    kappam : float
        Mean-reversion speed of long-term variance center (OU process)
    thetam : float
        Long-run mean of variance center
    omegam : float
        Volatility of variance center (OU diffusion coefficient)
    kappa1 : float
        Mean-reversion speed of instantaneous variance v1
    theta1 : float
        Long-run mean of instantaneous variance v1
    omega1 : float
        Vol-of-vol parameter for v1 process
    rho1 : float
        Correlation between log-VIX and v1 innovations
    bv : float
        Decay rate of shot noise for variance jumps
    lamb : float
        Jump arrival intensity (Poisson rate)
    muJV : float
        Mean jump size for variance (exponential distribution)
    """
    T: float
    kappa: float
    kappam: float
    thetam: float
    omegam: float
    kappa1: float
    theta1: float
    omega1: float
    rho1: float
    bv: float
    lamb: float
    muJV: float
    
    def to_list(self) -> List[float]:
        """Convert parameters to list format for legacy function compatibility."""
        return [self.T, self.kappa, self.kappam, self.thetam, self.omegam,
                self.kappa1, self.theta1, self.omega1, self.rho1,
                self.bv, self.lamb, self.muJV]


@dataclass
class InitialConditions:
    """
    Initial values for state variables in the reduced model.
    
    Attributes:
    -----------
    VIX0 : float
        Initial VIX level (not log-transformed)
    v10 : float
        Initial instantaneous variance
    Lv0 : float
        Initial shot noise component for variance
    m0 : float
        Initial long-term variance center
    """
    VIX0: float
    v10: float
    Lv0: float
    m0: float
    
    def to_list(self) -> List[float]:
        """Convert initial conditions to list format."""
        return [self.VIX0, self.v10, self.Lv0, self.m0]
