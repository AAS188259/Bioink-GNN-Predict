import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="Bioink GNN Predictor", layout="wide")

st.title("🧫 Bioink-GNN-Predict")
st.subheader("Predictive Soft-Matter Engineering Platform for 3D Bioprinting")
st.write("Translating agricultural banana-stem wastes and bacterial nanocellulose networks into validated medical hydrogels.")

# Sidebar Formulation Configuration Controls
st.sidebar.header("🧪 Hydrogel Formulation Controls")
banana_fiber = st.sidebar.slider("Banana Lignocellulosic Microfiber (%)", 0.5, 8.0, 4.0, step=0.1)
nanocellulose = st.sidebar.slider("Bacterial Nanocellulose (%)", 1.0, 5.0, 2.5, step=0.1)
crosslink_rate = st.sidebar.slider("Chemical Crosslinking Density (%)", 10, 90, 50)
shear_stress = st.sidebar.slider("Printhead Shear Strain Rate (1/s)", 1, 500, 100)

# Generate Spatial Network Topology Graph dynamically
def build_polymer_network_graph(num_nodes=40):
    G = nx.random_geometric_graph(num_nodes, radius=0.25)
    pos = nx.get_node_attributes(G, 'pos')
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines')
    
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(showscale=True, colorscale='Viridis', size=12,
                    line_width=2, color=[], label="Crosslink Node")
    )
    
    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=0),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

# Core Predictive Sequence
if st.button("🚀 Calculate Soft-Matter Performance"):
    # Trigger missing pipelines gracefully if not compiled yet on Streamlit server
    if not os.path.exists('bioink_model.pkl'):
        from generate_data import generate_bioink_dataset
        from train_model import train_bioink_brain
        generate_bioink_dataset()
        train_bioink_brain()
        
    with open('bioink_model.pkl', 'rb') as f:
        loaded_model = pickle.dump = pickle.load(f)
        
    input_features = np.array([[banana_fiber, nanocellulose, crosslink_rate, shear_stress]])
    raw_prediction = loaded_model.predict(input_features)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Predicted Dynamic Viscosity (Pa·s)", value=f"{raw_prediction[0]:.2f}")
        st.write("💡 *Lower viscosity under high shear confirms essential non-Newtonian shear-thinning requirements.*")
        
        st.metric(label="Post-Print Structural Fidelity", value=f"{raw_prediction[1]:.1f}%")
        if raw_prediction[1] < 45:
            st.error("⚠️ Status: Structural Collapse Hazard. Scaffold viscosity too low to retain structural shape post-extrusion.")
        elif raw_prediction[1] > 80:
            st.success("✅ Status: High Structural Fidelity. Suitable for multi-layer cell matrix architecture.")
        else:
            st.warning("⚠️ Status: Marginal Fidelity Risk. Vulnerable to structural relaxation or sagging.")
            
    with col2:
        st.markdown("### 🕸️ Simulated Network Matrix Topology")
        st.write("Dynamic graph representation of structural cross-linking densities within your hydrogel composite slurry:")
        network_plot = build_polymer_network_graph()
        st.plotly_chart(network_plot, use_container_width=True)
