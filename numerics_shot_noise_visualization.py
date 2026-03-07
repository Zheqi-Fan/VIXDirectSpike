"""
================================================================================
Shot Noise Visualization: Compound Poisson Process Comparison
================================================================================

This module generates comparative visualizations of shot noise processes with
different jump distributions to illustrate the impact of decaying jump dynamics.

The script produces a side-by-side comparison figure showing:
- Left panel: Normal jump distribution (both positive and negative magnitudes)
- Right panel: Exponential jump distribution (only positive magnitudes)

Both panels vary the mean-reversion decay rate b to demonstrate the effect on
shot noise trajectories.

Date: 2026
Purpose: Replication package
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use(['science', 'no-latex'])


def GeneratePathsEuler_Exponential(NoOfPaths, NoOfSteps, T, bs, muJ, eta, seed=10):
    """
    Generate paths with exponential jump distribution (only positive magnitude).
    
    This function simulates a shot noise process with exponentially distributed
    jump sizes and compound Poisson arrival process.
    
    Parameters
    ----------
    NoOfPaths : int
        Number of simulation paths
    NoOfSteps : int
        Number of discretization steps
    T : float
        Time horizon in years
    bs : float
        Mean-reversion decay rate (years^-1)
    muJ : float
        Mean of exponential jump distribution
    eta : float
        Jump arrival intensity (jumps per year)
    seed : int, optional
        Random seed for reproducibility (default: 10)
    
    Returns
    -------
    time : np.ndarray
        Time grid array of shape (NoOfSteps + 1,)
    Ls : np.ndarray
        Simulated paths of shape (NoOfPaths, NoOfSteps + 1)
    """
    np.random.seed(seed)
    time = np.zeros(NoOfSteps + 1)
    dt = T / float(NoOfSteps)
    
    # Jump terms - Exponential (only positive)
    ZPois = np.random.poisson(eta * dt, [NoOfPaths, NoOfSteps])
    J = np.random.exponential(muJ, [NoOfPaths, NoOfSteps])
    
    # Initialize shot noise process
    Ls = np.zeros([NoOfPaths, NoOfSteps + 1])
    Ls[:, 0] = 0
    
    # Euler discretization with exponential decay and Poisson jumps
    for i in range(0, NoOfSteps):
        Ls[:, i + 1] = Ls[:, i] - bs * Ls[:, i] * dt + J[:, i] * ZPois[:, i]
        time[i + 1] = time[i] + dt
    
    return time, Ls


def GeneratePathsEuler_Normal(NoOfPaths, NoOfSteps, T, bs, muJ, sigmaJ, eta, seed=10):
    """
    Generate paths with normal jump distribution (both positive and negative).
    
    This function simulates a shot noise process with normally distributed jump
    sizes and compound Poisson arrival process.
    
    Parameters
    ----------
    NoOfPaths : int
        Number of simulation paths
    NoOfSteps : int
        Number of discretization steps
    T : float
        Time horizon in years
    bs : float
        Mean-reversion decay rate (years^-1)
    muJ : float
        Mean of normal jump distribution
    sigmaJ : float
        Standard deviation of normal jump distribution
    eta : float
        Jump arrival intensity (jumps per year)
    seed : int, optional
        Random seed for reproducibility (default: 10)
    
    Returns
    -------
    time : np.ndarray
        Time grid array of shape (NoOfSteps + 1,)
    Ls : np.ndarray
        Simulated paths of shape (NoOfPaths, NoOfSteps + 1)
    """
    np.random.seed(seed)
    time = np.zeros(NoOfSteps + 1)
    dt = T / float(NoOfSteps)
    
    # Jump terms - Normal (both positive and negative)
    ZPois = np.random.poisson(eta * dt, [NoOfPaths, NoOfSteps])
    J = np.random.standard_normal([NoOfPaths, NoOfSteps])
    J = muJ + J * sigmaJ
    
    # Initialize shot noise process
    Ls = np.zeros([NoOfPaths, NoOfSteps + 1])
    Ls[:, 0] = 0
    
    # Euler discretization with exponential decay and Poisson jumps
    for i in range(0, NoOfSteps):
        Ls[:, i + 1] = Ls[:, i] - bs * Ls[:, i] * dt + J[:, i] * ZPois[:, i]
        time[i + 1] = time[i] + dt
    
    return time, Ls


def create_comparison_figure(output_dir='./'):
    """
    Create and save the comparative visualization of jump distributions.
    
    Parameters
    ----------
    output_dir : str, optional
        Output directory for saved figures (default: current directory)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    logger.info("Creating comparison visualization...")
    
    # Configuration parameters
    NoOfPaths, NoOfSteps, T = 10, 256, 2
    path_i = 3  # Index of path to display
    bs_list = [0, 5, 10, 50]
    
    # High-contrast scientific palette (colorblind-friendly)
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']  # Red, Blue, Green, Purple
    styles = ['dashed', 'dashdot', 'dotted', '-']
    
    # Jump distribution parameters
    muJ_exp, eta_exp = 0.3, 1.5
    muJ_norm, sigmaJ_norm, eta_norm = 0.05, 0.1, 1.5
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # =====================================================================
    # LEFT PANEL: Normal Jump Distribution
    # =====================================================================
    logger.info("Generating Normal jump distribution paths...")
    ax = axes[0]
    
    for bs, color, style in zip(bs_list, colors, styles):
        time, Ls = GeneratePathsEuler_Normal(
            NoOfPaths, NoOfSteps*T, T, bs, muJ_norm, sigmaJ_norm, eta_norm
        )
        ax.plot(time, Ls[path_i, :], label=f'$b$={bs}', 
                color=color, linestyle=style, linewidth=2)
    
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Shot Noise Level $L$', fontsize=11)
    ax.set_title('Normal Jump Distribution\n(Positive & Negative Magnitude)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # =====================================================================
    # RIGHT PANEL: Exponential Jump Distribution
    # =====================================================================
    logger.info("Generating Exponential jump distribution paths...")
    ax = axes[1]
    
    for bs, color, style in zip(bs_list, colors, styles):
        time, Ls = GeneratePathsEuler_Exponential(
            NoOfPaths, NoOfSteps*T, T, bs, muJ_exp, eta_exp
        )
        ax.plot(time, Ls[path_i, :], label=f'$b$={bs}', 
                color=color, linestyle=style, linewidth=2)
    
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Shot Noise Level $L$', fontsize=11)
    ax.set_title('Exponential Jump Distribution\n(Only Positive Magnitude)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'jump_distribution_comparison.eps'
    plt.savefig(output_path, dpi=400, format='eps')
    logger.info(f"✓ Figure saved: {output_path}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("📊 SIMULATION CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Simulation Parameters:")
    logger.info(f"  Time horizon (T):        {T} years")
    logger.info(f"  Number of paths:         {NoOfPaths}")
    logger.info(f"  Time steps:              {NoOfSteps*T}")
    logger.info(f"\nLeft Panel - Normal Jump Distribution:")
    logger.info(f"  Mean jump size (μ_J):    {muJ_norm}")
    logger.info(f"  Jump std dev (σ_J):      {sigmaJ_norm}")
    logger.info(f"  Jump intensity (η):      {eta_norm} jumps/year")
    logger.info(f"\nRight Panel - Exponential Jump Distribution:")
    logger.info(f"  Mean jump size (μ_J):    {muJ_exp}")
    logger.info(f"  Jump intensity (η):      {eta_exp} jumps/year")
    logger.info(f"\nDecay Rates (b):         {bs_list} years⁻¹")
    logger.info("="*70)
    
    return fig


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("Shot Noise Visualization - Comparative Analysis")
    logger.info("="*70)
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path('./')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comparison figure
        fig = create_comparison_figure(output_dir=output_dir)
        
        logger.info("\n✓ Visualization completed successfully!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        raise


if __name__ == '__main__':
    main()
