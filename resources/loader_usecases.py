import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import torch
import joblib
from resources.AIRMODE.load_helpers import load_mat_auto


def prepare_case(case_id, Nemp=20, Nsim=1_000, DGM=3, lb=None, ub =None):

    if case_id ==1:
        """ the paraboloid model with 3 data types"""

        if lb is None:
           lb = -5
        if ub is None:
            ub = +5
        if DGM==2:
            xi_list = np.array([0.0])
            M, Demp, Dsim = prepare_case_1_data(Nemp=Nemp, Nsim=Nsim, xi_list=xi_list, lb=lb, ub=ub)
        elif DGM==3:
            xi_list = np.array([-2.0, -1.0, 0.0, .2, 1.3, 2])
            M, Demp, Dsim = prepare_case_1_data(Nemp=Nemp, Nsim=Nsim, xi_list=xi_list, lb=lb, ub=ub)
        else: #DMG-1
            xi_list = np.array([0.0])
            Nemp=1
            M, Demp, Dsim = prepare_case_1_data(Nemp=Nemp, Nsim=Nsim, xi_list=xi_list, lb=lb, ub=ub)


    elif case_id == 2:
        """ the AIRMOD data"""
        M, Demp, Dsim = prepare_case_2_data(Nemp=Nemp, Nsim=Nsim)

    elif case_id == 3:
        """ the energy+ data set"""
        print('to be implemented')
        M, Demp, Dsim = [],[], []
    else:
        raise ValueError(f"Unknown case_id: {case_id}")

    return M, Demp, Dsim


def prepare_case_1_data(Nemp=20, Nsim=1_000, xi_list = np.array([-2.0, -1.0, 0.0, .2, 1.3, 2]), lb=None, ub=None, seed=123):
    """loading model M, empirical data Demp, and simulated archived Dsim for case 1 """
    if  lb is None:
        lb=-10
    if ub is None:
        ub = +10

    rng = np.random.default_rng(seed)

    def paraboloid_model(theta, xi=0.0):
        A, B, C = 1.0, 0.5, 1.5
        x1, x2 = theta[:, 0], theta[:, 1]
        xi = np.asarray(xi, dtype=float)
        return A * x1**2 + B * x1 * x2 * (1.0 + xi) + C * (x2 + xi) ** 2

    def sample_true_pdf_theta(Nemp):
        # Sample from multivariate Gaussian:
        # mean = [4.2, 3.2]  covariance = [[2.0, -0.7], [-0.7, 3.1]]
        mean = [2.0, 3.0]
        cov = [[0.5, 0.0], [0.0, 0.5]]
        return rng.multivariate_normal(mean, cov, size=Nemp)


    def generate_empirical_data(Nemp, xi_list):
        data_empirical = {}
        for i, xi in enumerate(xi_list):
            theta = sample_true_pdf_theta(Nemp)
            y_data = paraboloid_model(theta, xi=xi)
            data_empirical[i] = {"xi": xi,
                                 "theta": theta,
                                 "y_data": y_data
            }
        return data_empirical


    def generate_simulated_data(Nsim, xi_list):
        data_simulated = {}
        theta = rng.uniform(lb, ub, size=(Nsim, 2))
        for i, xi in enumerate(xi_list):
            y_data = paraboloid_model(theta, xi=xi)
            data_simulated[i] = {"xi": xi,
                "theta": theta,
                "y_data": y_data
            }
        return data_simulated

    Model = paraboloid_model
    Demp = generate_empirical_data(Nemp=Nemp, xi_list=xi_list)
    Dsim = generate_simulated_data(Nsim=Nsim, xi_list=xi_list)
    return Model, Demp, Dsim



def prepare_case_2_data(Nemp=20,
                        Nsim=1_000,
                        sim_data_path="../resources/AIRMODE/airmod_io_repo_style_500k.csv",
                        emp_data_path="../resources/AIRMODE/data/DLRAirmodData.mat",
                        model_path="AIRMODE/airmode_surrogate.pt",
                        scaler_path="AIRMODE/airmode_y_scaler.pkl",
                        ):
    """loading model M, empirical data Demp, and simulated archived Dsim for case 2 """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------------------------------------
    # 1) Load surrogate model M
    checkpoint = torch.load(model_path, map_location=device)
    y_scaler = joblib.load(scaler_path)

    model = MLPRegressor(
        in_dim=checkpoint["in_dim"],
        out_dim=checkpoint["out_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully")
    print("Expected input columns:", checkpoint["theta_cols"])
    print("Output columns:", checkpoint["y_cols"])

    def Model(X_new):
        X_new = np.asarray(X_new, dtype=np.float32)
        if X_new.ndim == 1:
            X_new = X_new[None, :]

        with torch.no_grad():
            X_t = torch.tensor(X_new, dtype=torch.float32).to(device)
            Y_pred_scaled = model(X_t).cpu().numpy()

        Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
        return Y_pred

    # --------------------------------------------------
    # 2) Load empirical data Demp
    m = load_mat_auto(emp_data_path)
    emp_key = "artificialDataWide" if "artificialDataWide" in m else "artificialData"
    Y_emp = np.array(m[emp_key], dtype=np.float32)
    if Nemp is not None:
        Y_emp = Y_emp[:Nemp]
    Demp = { 0: {"y_data": Y_emp}}
    print(f"[EMPIRICAL] Y_emp (from {emp_data_path}) -> {Y_emp.shape}")

    # --------------------------------------------------
    # 3) Load simulated data Dsim
    df = pd.read_csv(sim_data_path)
    theta_cols = checkpoint["theta_cols"]
    y_cols = checkpoint["y_cols"]
    X_sim = df[theta_cols].to_numpy(dtype=np.float32)
    Y_sim = df[y_cols].to_numpy(dtype=np.float32)
    if Nsim is not None:
        X_sim = X_sim[:Nsim]
        Y_sim = Y_sim[:Nsim]
    Dsim = { 0: {"theta": X_sim, "y_data": Y_sim} }
    print(f"[SIMULATED] X_sim -> {X_sim.shape}")
    print(f"[SIMULATED] Y_sim -> {Y_sim.shape}")
    return Model, Demp, Dsim



import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class MLPRegressor(nn.Module):
    def __init__(self, in_dim=11, out_dim=10, hidden_dims=(128, 128, 64), dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

