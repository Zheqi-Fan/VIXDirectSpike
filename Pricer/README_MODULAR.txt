VIX Option Pricing - Reduced Model (Modular Structure)
======================================================

This directory contains a modular implementation of the reduced VIX shot noise model
for option pricing. The model excludes level jumps (bs=0, Ls0=0, muJS not used).

File Structure
--------------

1. model_config.py
   - ModelParameters: dataclass with all model parameters
   - InitialConditions: dataclass with initial state values

2. characteristic_function.py
   - CharacteristicFunctionSolver: solves ODEs for affine coefficients
   - Methods: coef_A, coef_B, coef_D, coef_E, coef_F
   - compute_characteristic_function()

3. fft_pricing.py
   - FFTPricer: Carr-Madan FFT method for option pricing
   - price_call_fft(): full FFT pricing
   - price_single_call(): single strike pricing

4. monte_carlo.py
   - MonteCarloSimulator: path simulation using Euler-Maruyama
   - generate_paths(): simulate VIX paths
   - price_call_mc(): Monte Carlo option pricing

5. validator.py
   - ModelValidator: validation framework
   - validate_pricing(): compare FFT vs MC across strikes

6. run_validation.py
   - main(): validation test script
   - Run this file to execute the validation

Usage
-----

To run the validation:
    python run_validation.py

To import modules in your own code:
    from model_config import ModelParameters, InitialConditions
    from fft_pricing import FFTPricer
    from monte_carlo import MonteCarloSimulator

Dependencies
------------
- numpy
- pandas
- scipy
- tqdm
- toolkit (custom module with odeRK4)

Model Features
--------------
- Variance jumps only
- OU process for long-term variance center
- CIR-type stochastic volatility
- Shot noise for variance jumps (Lv)