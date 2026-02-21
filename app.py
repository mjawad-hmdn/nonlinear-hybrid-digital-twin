import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import pickle
import time

# ==============================================
# Load Nonlinear Model + Normalization
# ==============================================

class NonlinearAccelerationTwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = NonlinearAccelerationTwin()
model.load_state_dict(torch.load("nonlinear_acc_model.pt"))
model.eval()

with open("nonlinear_normalization.pkl", "rb") as f:
    norm_data = pickle.load(f)

X_mean = norm_data["X_mean"]
X_std = norm_data["X_std"]
y_mean = norm_data["y_mean"]
y_std = norm_data["y_std"]

# ==============================================
# Nonlinear Physics Solver
# ==============================================

def physics_solver(m, k, c, alpha, u0, v0, dt, t_final):

    n_steps = int(t_final / dt)
    time_arr = np.linspace(0, t_final, n_steps)

    u = np.zeros(n_steps)
    v = np.zeros(n_steps)

    u[0] = u0
    v[0] = v0

    for i in range(n_steps - 1):
        a = (-c*v[i] - k*u[i] - alpha*(u[i]**3)) / m
        v[i+1] = v[i] + dt * a
        u[i+1] = u[i] + dt * v[i]

    return time_arr, u

# ==============================================
# Nonlinear Hybrid Solver
# ==============================================
def hybrid_solver_mc(m, k, c, alpha, u0, v0, dt, t_final, n_samples=30):

    n_steps = int(t_final / dt)
    time_arr = np.linspace(0, t_final, n_steps)

    all_trajectories = []

    model.train()  # activate dropout

    for s in range(n_samples):

        u = np.zeros(n_steps)
        v = np.zeros(n_steps)

        u[0] = u0
        v[0] = v0

        for i in range(n_steps - 1):

            x_input = np.array([[m, k, c, alpha, u[i], v[i]]])
            x_norm = (x_input - X_mean) / X_std
            x_tensor = torch.tensor(x_norm, dtype=torch.float32)

            with torch.no_grad():
                a_norm = model(x_tensor).numpy()

            a = (a_norm * y_std + y_mean)[0][0]

            v[i+1] = v[i] + dt * a
            u[i+1] = u[i] + dt * v[i]

        all_trajectories.append(u)

    model.eval()

    all_trajectories = np.array(all_trajectories)

    mean_traj = np.mean(all_trajectories, axis=0)
    std_traj = np.std(all_trajectories, axis=0)

    return time_arr, mean_traj, std_traj, all_trajectories



# ==============================================
# Streamlit UI
# ==============================================

st.title("Nonlinear Hybrid Digital Twin Platform")

st.sidebar.header("System Parameters")

m = st.sidebar.number_input("Mass (m)", 0.1, 10.0, 2.0)
k = st.sidebar.number_input("Stiffness (k)", 10.0, 1000.0, 200.0)
c = st.sidebar.number_input("Damping (c)", 0.0, 10.0, 1.0)
alpha = st.sidebar.number_input("Nonlinear Coefficient (alpha)", -1000.0, 1000.0, 0.0)
u0 = st.sidebar.number_input("Initial Displacement", -1.0, 1.0, 0.2)
v0 = st.sidebar.number_input("Initial Velocity", -5.0, 5.0, 0.0)
dt = st.sidebar.number_input(
    "Time Step (dt)",
    min_value=0.00001,
    max_value=0.1,
    value=0.001,
    step=0.0001,
    format="%.5f"
)
t_final = st.sidebar.number_input(
    "Simulation Time",
    min_value=0.1,
    max_value=30.0,
    value=5.0,
    step=0.1
)

if st.button("Run Simulation"):

    # ===== Physics Run =====
    start_phys = time.time()
    time_phys, u_phys = physics_solver(m, k, c, alpha, u0, v0, dt, t_final)
    phys_time = time.time() - start_phys

    # ===== Hybrid Monte Carlo Run =====
    start_hyb = time.time()
    time_hyb, mean_traj, std_traj, samples = hybrid_solver_mc(
        m, k, c, alpha, u0, v0, dt, t_final, n_samples=30
    )
    hyb_time = time.time() - start_hyb

    # ===== RMSE (mean vs physics) =====
    rmse = np.sqrt(np.mean((u_phys - mean_traj)**2))

    # ===== Plot =====
    fig = go.Figure()

    # Physics
    fig.add_trace(go.Scatter(
        x=time_phys,
        y=u_phys,
        mode='lines',
        name='Physics'
    ))

    # Hybrid Mean
    fig.add_trace(go.Scatter(
        x=time_hyb,
        y=mean_traj,
        mode='lines',
        name='Hybrid Mean',
        line=dict(color='red')
    ))

    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_hyb, time_hyb[::-1]]),
        y=np.concatenate([mean_traj - std_traj,
                          (mean_traj + std_traj)[::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1σ'
    ))

    # Sample trajectories (show 5)
    for s in range(5):
        fig.add_trace(go.Scatter(
            x=time_hyb,
            y=samples[s],
            mode='lines',
            line=dict(width=1, dash='dot'),
            showlegend=False
        ))

    fig.update_layout(
        title="Probabilistic Nonlinear Hybrid Twin",
        xaxis_title="Time",
        yaxis_title="Displacement",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===== Metrics =====
    st.subheader("Performance Metrics")
    st.write(f"RMSE (mean vs physics): {rmse:.6f}")
    st.write(f"Physics Runtime: {phys_time:.6f} sec")
    st.write(f"Hybrid Runtime: {hyb_time:.6f} sec")

    if hyb_time > 0:
        speed_ratio = phys_time / hyb_time
        st.write(f"Speed Ratio (Physics/Hybrid): {speed_ratio:.2f}")

