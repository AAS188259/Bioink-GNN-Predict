import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def execute_pipeline():
    data_path = 'bioink_simulated_data.csv'
    if not os.path.exists(data_path):
        from generate_data import generate_rheology_matrix
        generate_rheology_matrix()
        
    df = pd.read_csv(data_path)
    
    features = ['c_banana', 'c_bnc', 'rho_xl', 'gamma_dot']
    targets = ['viscosity', 'fidelity']
    
    X = df[features]
    y = df[targets]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # High-dimensional non-linear mapping engine
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    with open('bioink_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    execute_pipeline() 
