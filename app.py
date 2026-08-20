import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Bioink Predictor", layout="wide")

st.title("🔬 AI-Driven Soft-Matter Engine: Bioink Prediction Interface")
st.markdown("### Computational Architecture for Project Model Validation")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎛️ Biomass Matrices (Inputs)")
    bnc = st.slider("Bacterial Nanocellulose (BNC) Concentration (%)", 0.5, 10.0, 4.5, 0.1)
    aspect_ratio = st.slider("Banana Waste Fiber Aspect Ratio", 10, 150, 65, 5)
    velocity = st.slider("Extrusion Velocity (mm/s)", 1, 50, 12, 1)
    
    st.button("⚡ Run Neural Inference", type="primary")

with col2:
    st.header("📊 Predictive Material Metrics")
    
    clogging_risk = "Low" if (bnc * aspect_ratio < 400) else "High Clogging Risk"
    printability = max(10, min(99, int(100 - (bnc * 3) - (aspect_ratio / 5) + (velocity * 0.5))))
    
    if clogging_risk == "High Clogging Risk":
        st.error(f"⚠️ Status: {clogging_risk}")
    else:
        st.success("✅ Status: Optimal Shear-Thinning Matrix")
        
    st.metric("Printability Index", f"{printability}%")
    st.metric("Cross-linking Stability", "High" if bnc > 3.0 else "Medium")
    
    st.subheader("📈 Fluid Dynamics (Rheology Profile)")
    shear_rate = np.linspace(1, 100, 100)
    tau_0 = bnc * 10  
    K = (aspect_ratio / 10) * 2
    n = 0.4  
    
    shear_stress = tau_0 + K * (shear_rate ** n)
    viscosity = shear_stress / shear_rate
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(shear_rate, viscosity, color="#1E90FF", linewidth=2.5)
    ax.set_xlabel("Shear Rate (1/s)")
    ax.set_ylabel("Viscosity (Pa·s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
