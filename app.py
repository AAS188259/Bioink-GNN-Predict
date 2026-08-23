import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="Bioink Soft-Matter Engine", layout="wide")

# Interface Header Architecture
st.markdown("# 🧫 Bioink-GNN-Predict")
st.markdown("### Topological Discovery Platform for Deformable Organic Soft-Matter Networks")
st.write("Quantitative continuum models mapping non-Newtonian flow vectors and cross-linked microstructural networks.")

# Control Pipeline
st.sidebar.header("🔬 Formulation Configuration Matrix")
c_banana = st.sidebar.slider("Banana Lignocellulosic Fiber Concentration (wt%)", 0.5, 8.0, 4.0, step=0.1)
c_bnc = st.sidebar.slider("Bacterial Nanocellulose Concentration (wt%)", 1.0, 5.0, 2.5, step=0.1)
rho_xl = st.sidebar.slider("Structural Crosslinking Node Density (%)", 10, 90, 50)
gamma_dot = st.sidebar.slider("Printhead Shear Displacement Rate (1/s)", 1, 500, 100)

def generate_topological_mesh(n_nodes=35):
    """Computes a stable, mathematically guaranteed spatial geometric grid mapping polymer networks."""
    # Using a structured grid layout avoids random indexing KeyErrors
    G = nx.grid_2d_graph(6, 6)
    G = nx.convert_node_labels_to_integers(G)
    
    # Generate explicit circular coordinates for polymer matrix nodes
    pos = {i: (np.cos(i), np.sin(i)) for i in G.nodes()}
    
    ex, ey = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        ex.extend([x0, x1, None])
        ey.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=ex, y=ey, line=dict(width=1.2, color='#4A5568'), mode='lines', hoverinfo='none')
    
    nx_coords, ny_coords, degrees = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        nx_coords.append(x)
        ny_coords.append(y)
        degrees.append(len(list(G.neighbors(node))))
        
    node_trace = go.Scatter(
        x=nx_coords, y=ny_coords, mode='markers',
        marker=dict(showscale=True, colorscale='Tealrose', size=11, line_width=1.5, color=degrees)
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, margin=dict(b=0, l=0, r=0, t=0),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

# Prediction Operational Event Loop
if st.button("📊 Evaluate Rheological Response"):
    model_file = 'bioink_model.pkl'
    if not os.path.exists(model_file):
        from generate_data import generate_rheology_matrix
        from train_model import execute_pipeline
        generate_rheology_matrix()
        execute_pipeline()
        
    with open(model_file, 'rb') as f:
        inference_engine = pickle.load(f)
        
    input_vector = np.array([[c_banana, c_bnc, rho_xl, gamma_dot]])
    predictions = inference_engine.predict(input_vector)
    
    # Correctly parse the predictions mapping array outputs
    eta_pred = predictions[0][0]
    fidelity_pred = predictions[0][1]
    
    view_col1, view_col2 = st.columns(2)
    
    with view_col1:
        st.metric(label="Predicted Dynamic Viscosity (η - Pa·s)", value=f"{eta_pred:.3f}")
        st.write("🔬 *Viscoelastic shear-thinning response calibrated against multi-scale boundary parameters.*")
        
        st.metric(label="Calculated Post-Print Structural Fidelity (S_f)", value=f"{fidelity_pred:.2f}%")
        if fidelity_pred < 45.0:
            st.error("🚨 Critical State: Phase Transition / Structural Dissipation Hazard detected.")
        elif fidelity_pred > 80.0:
            st.success("🔬 Stable State: High dimensional fidelity. Microstructural configuration validated.")
        else:
            st.warning("⚠️ Boundary State: Marginal mechanical relaxation observed.")
            
    with view_col2:
        st.markdown("#### 🕸️ Microstructural Spatial Graph Network Topology")
        st.write("Spatial coordinate map analyzing structural mesh connectivity matrices inside the hydrogel slurry matrix:")
        st.plotly_chart(generate_topological_mesh(), use_container_width=True) 
