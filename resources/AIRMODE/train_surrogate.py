# retrain_airmod.py
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# --- assume you already have these in memory; otherwise read CSV then:
# import pandas as pd; df = pd.read_csv("airmod_io_repo_style.csv")
# theta_scaled_cols = [c for c in df.columns if c.endswith("_scaled")]
# y_cols = ['D1_roll','D2_pitch','D3_heave','D4_2ndEF_bend','D5_3rdEF_bend',
#           'D6_antisym_tors','D7_sym_tors','D8_vtp_bend','D9_4thEF_bend','D10_1stEF_foreaft']
# X_scaled = df[theta_scaled_cols].to_numpy(np.float32)
# Y_sim    = df[y_cols].to_numpy(np.float32)


# ---- Python inference convenience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_surrogate(X_sim, Y_sim):
    X = torch.tensor(X_sim, dtype=torch.float32)
    Y = torch.tensor(Y_sim,    dtype=torch.float32)

    # Split
    X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_va, X_te, Y_va, Y_te   = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42)

    # Simple output standardization (helps training; invert after inference)
    y_mean = Y_tr.mean(0)
    y_std  = Y_tr.std(0).clamp_min(1e-6)
    Y_tr_z, Y_va_z, Y_te_z = (Y_tr - y_mean)/y_std, (Y_va - y_mean)/y_std, (Y_te - y_mean)/y_std

    bs = 2048
    dl_tr = DataLoader(TensorDataset(X_tr, Y_tr_z), batch_size=bs, shuffle=True, drop_last=False)
    dl_va = DataLoader(TensorDataset(X_va, Y_va_z), batch_size=bs, shuffle=False)
    dl_te = DataLoader(TensorDataset(X_te, Y_te_z), batch_size=bs, shuffle=False)

    # --- Model (replicates your MATLAB topology)
    class AirmodMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn0  = nn.BatchNorm1d(11)
            self.fc1  = nn.Linear(11, 60)
            self.bn1  = nn.BatchNorm1d(60)
            self.fc2  = nn.Linear(60, 60)
            self.bn2  = nn.BatchNorm1d(60)
            self.fc3  = nn.Linear(60, 60)
            self.fc4  = nn.Linear(60, 10)  # linear head

        def forward(self, x):
            x = self.bn0(x); x = torch.tanh(self.fc1(x))
            x = self.bn1(x); x = torch.tanh(self.fc2(x))
            x = self.bn2(x); x = torch.tanh(self.fc3(x))
            x = self.fc4(x)
            return x

        def airmod_predict_py(self, theta_scaled_np: np.ndarray) -> np.ndarray:
            x = torch.as_tensor(theta_scaled_np, dtype=torch.float32, device=device)
            if x.ndim == 1: x = x[None, :]
            with torch.no_grad():
                y_z = self(x)
                y = (y_z.cpu() * y_std + y_mean).numpy()
            return y


    model = AirmodMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=False)
    loss_fn = nn.MSELoss()

    best = {'val': float('inf'), 'state': None}
    patience, patience_ctr = 20, 0
    max_epochs = 200

    for epoch in range(1, max_epochs+1):
        # train
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(dl_tr.dataset)

        # validate
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                va_loss += loss_fn(pred, yb).item() * xb.size(0)
        va_loss /= len(dl_va.dataset)
        sched.step(va_loss)

        # early stopping
        if va_loss < best['val'] - 1e-6:
            best['val'], best['state'] = va_loss, model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience: break

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train {tr_loss:.5f} | val {va_loss:.5f}")

    # load best and evaluate on test
    model.load_state_dict(best['state'])
    model.eval()
    def eval_rmse(loader):
        se, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                pred_z = model(xb)
                pred = pred_z.cpu()*y_std + y_mean  # unscale back to original Y units
                y = yb if yb.ndim==2 else yb[0]
                se += ((pred - y)**2).sum(0)
                n += y.shape[0]
        rmse = torch.sqrt(se / n)
        return rmse.numpy()

    rmse = eval_rmse(dl_te)
    print("Test RMSE per output:", rmse)

    # ---- Save model + scalers
    torch.save({
        'state_dict': model.state_dict(),
        'y_mean': y_mean.numpy(),
        'y_std': y_std.numpy()
    }, "airmod_mlp.pt")
    print("Saved airmod_mlp.pt")

    # ---- Export ONNX (optional)
    dummy = torch.randn(1, 11, dtype=torch.float32).to(device)
    torch.onnx.export(
        model, dummy, "airmod_mlp.onnx",
        input_names=["theta_scaled"], output_names=["y_scaled_z"],
        dynamic_axes={"theta_scaled": {0: "batch"}, "y_scaled_z": {0: "batch"}},
        opset_version=17
    )
    print("Saved airmod_mlp.onnx")

