import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_bioink_brain():
    # Load dataset
    df = pd.read_csv('bioink_simulated_data.csv')
    
    X = df[['banana_fiber_pct', 'bacterial_cellulose_pct', 'crosslink_density', 'shear_rate']]
    y = df[['predicted_viscosity', 'structural_fidelity']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)
    
    # Train predictor model mimicking network topological adjustments
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save trained ML engine to disk
    with open('bioink_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Physics prediction engine trained and successfully saved!")

if __name__ == "__main__":
    import os
    if not os.path.exists('bioink_simulated_data.csv'):
        from generate_data import generate_bioink_dataset
        generate_bioink_dataset()
    train_bioink_brain()

