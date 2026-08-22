import numpy as np
import pandas as pd

def generate_bioink_dataset(num_samples=200):
    np.random.seed(42)
    
    # Input Features (What a formulation scientist controls)
    banana_fiber_pct = np.random.uniform(0.5, 8.0, num_samples)      # 0.5% to 8% concentration
    bacterial_cellulose_pct = np.random.uniform(1.0, 5.0, num_samples) # 1% to 5% concentration
    crosslink_density = np.random.uniform(10, 90, num_samples)         # Cross-linking percentage
    shear_rate = np.random.uniform(1, 500, num_samples)                # Printhead shear strain (1/s)
    
    # Target Physics Metrics calculated using simulated continuum mechanics
    # 1. Non-Newtonian Shear Thinning Viscosity (Power-law fluid logic)
    viscosity = (100.0 * (banana_fiber_pct * 1.5) + (bacterial_cellulose_pct * 2.0)) / (shear_rate ** 0.4)
    viscosity += np.random.normal(0, viscosity * 0.05, num_samples) # Add real-world experimental noise
    
    # 2. Structural Printing Fidelity % (Higher fiber and crosslinking = stronger shape hold)
    base_fidelity = (banana_fiber_pct * 5) + (bacterial_cellulose_pct * 4) + (crosslink_density * 0.5)
    structural_fidelity = np.clip(base_fidelity - (viscosity * 0.1), 10, 98) 
    
    df = pd.DataFrame({
        'banana_fiber_pct': banana_fiber_pct,
        'bacterial_cellulose_pct': bacterial_cellulose_pct,
        'crosslink_density': crosslink_density,
        'shear_rate': shear_rate,
        'predicted_viscosity': viscosity,
        'structural_fidelity': structural_fidelity
    })
    
    df.to_csv('bioink_simulated_data.csv', index=False)
    print("✅ Successfully generated simulated soft-matter rheology dataset!")

if __name__ == "__main__":
    generate_bioink_dataset()
