import numpy as np
import pandas as pd

def generate_rheology_matrix(n_samples=250, seed=42):
    """
    Simulates non-Newtonian continuum mechanics for composite biopolymer hydrogels.
    Models structural network interactions of BNC and lignocellulosic matrices.
    """
    rng = np.random.default_rng(seed)
    
    # Input parameter matrix (Formulation constraints)
    C_banana = rng.uniform(0.5, 8.0, n_samples)      # Lignocellulosic concentration (%)
    C_bnc = rng.uniform(1.0, 5.0, n_samples)         # Bacterial nanocellulose concentration (%)
    rho_xl = rng.uniform(10.0, 90.0, n_samples)      # Crosslinking junction density (%)
    gamma_dot = rng.uniform(1.0, 500.0, n_samples)   # Extrusion shear strain rate (1/s)
    
    # Mathematical modeling of shear-thinning fluid dynamics (Ostwald-de Waele logic)
    n_index = 1.0 - (0.05 * C_banana + 0.03 * C_bnc)
    K_consistency = 15.0 * (C_banana ** 1.3) + 22.0 * (C_bnc ** 1.1)
    
    # Apparent Viscosity calculations: eta = K * (gamma_dot) ^ (n - 1)
    viscosity = K_consistency * (gamma_dot ** (n_index - 1.0))
    viscosity += rng.normal(0.0, viscosity * 0.04, n_samples)   # Stochastic experimental variance
    
    # Structural Fidelity parameter modeling post-extrusion structural yield stress
    structural_yield = (C_banana * 6.2) + (C_bnc * 4.8) + (rho_xl * 0.45) - (viscosity * 0.08)
    fidelity = np.clip(structural_yield, 5.0, 99.5)
    
    matrix_df = pd.DataFrame({
        'c_banana': C_banana,
        'c_bnc': C_bnc,
        'rho_xl': rho_xl,
        'gamma_dot': gamma_dot,
        'viscosity': viscosity,
        'fidelity': fidelity
    })
    
    matrix_df.to_csv('bioink_simulated_data.csv', index=False)
    return True

if __name__ == "__main__":
    generate_rheology_matrix() 
