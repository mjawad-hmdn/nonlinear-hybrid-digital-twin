import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Nonlinear Hybrid Digital Twin", layout="wide")
st.title("1-DOF Nonlinear Hybrid Oscillation Simulation")
st.markdown("Physics-based solver vs Machine Learning Hybrid model")

# =========================
# MODEL ARCHITECTURE
# (MATCHES TRAINED CHECKPOINT)
# =========================

class AccelerationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = AccelerationModel()
model.load_state_dict(
    torch.load("nonlinear_acc_model_dropout.pt", map_location=torch.device("cpu"))
)
model.eval()

# Load scaler
scaler = joblib.load("nonlinear_normalization.pkl")

# =========================
# SIDEBAR PARAMETERS
# =========================

st.sidebar.header("System Parameters")

m = st.sidebar.number_input("Mass (m)", value=2.0)
k = st.sidebar.number_input("Stiffness (k)", value=200.0)
c = st.sidebar.number_input("Damping (c)", value=1.0)
alpha = st.sidebar.number_input("Nonlinear Coefficient (alpha)", value=0.0)

u0 = st.sidebar.number_input("Initial Displacement", value=0.2)
v0 = st.sidebar.number_input("Initial Velocity", value=0.0)

t_final = st.sidebar.number_input("Simulation Time", value=5.0)

# =========================
# PHYSICS SOLVER
# =========================

def physics_solver(m, k, c, alpha, u0, v0, dt, t_final):
    t = np.arange(0, t_final, dt)
    n = len(t)

    u = np.zeros(n)
    v = np.zeros(n)

    u[0] = u0
    v[0] = v0

    for i in range(n - 1):
        a = (-c*v[i] - k*u[i] - alpha*u[i]**3) / m
        v[i+1] = v[i] + a * dt
        u[i+1] = u[i] + v[i] * dt

    return t, u

# =========================
# HYBRID SOLVER (DETERMINISTIC)
# =========================

def hybrid_solver(m, k, c, alpha, u0, v0, dt, t_final):
    t = np.arange(0, t_final, dt)
    n = len(t)

    u = np.zeros(n)
    v = np.zeros(n)

    u[0] = u0
    v[0] = v0

    for i in range(n - 1):
        inp = np.array([[m, k, c, alpha, u[i], v[i]]])
        X_mean = scaler["X_mean"]
        X_std = scaler["X_std"]

        inp_norm = (inp - X_mean) / X_std
        inp_tensor = torch.tensor(inp_norm, dtype=torch.float32)

        with torch.no_grad():
            a_pred = model(inp_tensor).item()

        v[i+1] = v[i] + a_pred * dt
        u[i+1] = u[i] + v[i] * dt

    return t, u

# =========================
# RUN SIMULATION
# =========================

# ===============================
# Run Simulation
# ===============================

if st.button("Run Simulation"):

    # ---------------------------------
    # Automatic Stability Handling
    # ---------------------------------
    # ---------------------------------
    # Automatic Stable Time Step
    # ---------------------------------
    omega_est = np.sqrt(k / m)
    dt = 1.0 / (20 * omega_est)  # 20 steps per oscillation

    # ---------------------------------
    # Run Physics Solver
    # ---------------------------------
    start_phys = time.time()
    t_phys, u_phys = physics_solver(m, k, c, alpha, u0, v0, dt, t_final)
    physics_time = time.time() - start_phys

    # ---------------------------------
    # Run Hybrid Solver
    # ---------------------------------
    start_hyb = time.time()
    t_hyb, u_hyb = hybrid_solver(m, k, c, alpha, u0, v0, dt, t_final)
    hybrid_time = time.time() - start_hyb

    # ---------------------------------
    # Compute Metrics
    # ---------------------------------
    rmse = np.sqrt(np.mean((u_phys - u_hyb) ** 2))
    speed_ratio = physics_time / hybrid_time if hybrid_time > 0 else np.nan

    # ---------------------------------
    # Plot Results
    # ---------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_phys, y=u_phys, name="Physics"))
    fig.add_trace(go.Scatter(x=t_hyb, y=u_hyb, name="Hybrid"))

    fig.update_layout(
        title="Physics vs Hybrid Digital Twin",
        xaxis_title="Time",
        yaxis_title="Displacement",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # Display Performance Metrics
    # ---------------------------------
    st.subheader("Performance Metrics")
    st.write(f"RMSE: {rmse:.6f}")
    st.write(f"Physics Runtime: {physics_time:.6f} sec")
    st.write(f"Hybrid Runtime: {hybrid_time:.6f} sec")
    st.write(f"Speed Ratio (Physics/Hybrid): {speed_ratio:.2f}")