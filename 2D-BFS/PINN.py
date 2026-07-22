#!/usr/bin/env python
# coding: utf-8
"""
PINN Training Script
Changes vs previous version:
  - phys_out points now have their OWN 4 physics losses (cont, x-mom, y-mom, adv)
    computed via a separate loader, NOT merged with phys_in.
  - Each phys_out loss is weighted by:   alpha * Lambda_X
    where alpha is a scalar in [0, 1]:
      alpha = 0  →  phys_out losses are zeroed out (no effect on training)
      alpha = 1  →  phys_out losses have equal weight to phys_in losses
  - 8 physics losses are tracked and logged (4 phys_in + 4 phys_out).
  - phys_out_frac controls the SIZE of the phys_out point set (unchanged semantics).
"""
Flag_schedule = True
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import vtk
from vtk.util import numpy_support as VN
from matplotlib import pyplot as plt
from datetime import datetime
import json
import re
import sys
torch.cuda.empty_cache()
# =============================================================================
# 0) HELPERS
# =============================================================================
def make_yregion_prefix(y1_min, y1_max, y2_min, y2_max):
    y1 = f"y1_{fmt_float_tag(y1_min)}_{fmt_float_tag(y1_max)}"
    if (y2_min is None) or (y2_max is None):
        y2 = "y2_None"
    else:
        y2 = f"y2_{fmt_float_tag(y2_min)}_{fmt_float_tag(y2_max)}"
    return f"{y1}__{y2}"
def make_time_run_dir(root="Runs"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = os.path.join(root, timestamp)
    os.makedirs(run_root, exist_ok=True)
    return run_root, timestamp
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
def fmt_float_tag(x):
    if x is None:
        return "None"
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")
def make_run_tag(file_start, file_end, file_step, Re, Pe,
                 y1_min, y1_max, y2_min, y2_max,
                 data_frac, phys_frac):
    y1 = f"y1_{fmt_float_tag(y1_min)}_{fmt_float_tag(y1_max)}"
    if (y2_min is None) or (y2_max is None):
        y2 = "y2_None"
    else:
        y2 = f"y2_{fmt_float_tag(y2_min)}_{fmt_float_tag(y2_max)}"
    return (f"start{file_start}_end{file_end}_step{file_step}_Re{Re}_Pe{Pe}_"
            f"{y1}_{y2}_df{fmt_float_tag(data_frac)}_pf{fmt_float_tag(phys_frac)}")
def build_run_dirs(run_root: str):
    run_root = ensure_dir(run_root)
    dirs = {
        "root":    run_root,
        "weights": ensure_dir(os.path.join(run_root, "weights")),
        "losses":  ensure_dir(os.path.join(run_root, "losses")),
        "results": ensure_dir(os.path.join(run_root, "results")),
        "plots":   ensure_dir(os.path.join(run_root, "plots")),
        "wss":     ensure_dir(os.path.join(run_root, "wss_results")),
    }
    return dirs
def apply_two_region_y_mask(t, x, y, T, Y_scale,
                             y1_min, y1_max, y2_min, y2_max, verbose=True):
    y_phys = y * Y_scale
    m1 = (y_phys >= y1_min) & (y_phys <= y1_max)
    if (y2_min is None) or (y2_max is None):
        mask = m1
        n2 = 0
    else:
        m2 = (y_phys >= y2_min) & (y_phys <= y2_max)
        mask = m1 | m2
        n2 = int(np.count_nonzero(m2))
    n_all  = y.shape[0]
    n_keep = int(np.count_nonzero(mask))
    n1     = int(np.count_nonzero(m1))
    if verbose:
        print(f"[MASK] Total points: {n_all}")
        print(f"[MASK] Region1 keep: {n1} (y in [{y1_min}, {y1_max}])")
        if (y2_min is None) or (y2_max is None):
            print("[MASK] Region2 disabled")
        else:
            print(f"[MASK] Region2 keep: {n2} (y in [{y2_min}, {y2_max}])")
        print(f"[MASK] Kept after OR: {n_keep} ({100.0*n_keep/max(1,n_all):.3f}%)")
    if n_keep < 10:
        raise RuntimeError(
            f"Too few data points after masking (kept={n_keep}). "
            "Check y1/y2 ranges and Y_scale."
        )
    return t[mask], x[mask], y[mask], T[mask]
# =============================================================================
# 1) VTK UTILITIES
# =============================================================================
def load_vtk_file(filename):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()
def extract_coordinates(data):
    n_points = data.GetNumberOfPoints()
    return np.array([data.GetPoint(i) for i in range(n_points)])
def extract_field_array(data, field_name):
    array = data.GetPointData().GetArray(field_name)
    if array is None:
        raise ValueError(f"Field '{field_name}' not found in VTK file.")
    return VN.vtk_to_numpy(array)
def load_wall_points(wall_filename, X_scale, Y_scale, device):
    if not os.path.exists(wall_filename):
        raise FileNotFoundError(f"Wall file not found: {wall_filename}")
    wall_data = load_vtk_file(wall_filename)
    coords = extract_coordinates(wall_data)
    xw = coords[:, 0:1] / X_scale
    yw = coords[:, 1:2] / Y_scale
    xw_t = torch.tensor(xw, dtype=torch.float32, device=device)
    yw_t = torch.tensor(yw, dtype=torch.float32, device=device)
    return xw_t, yw_t
# =============================================================================
# 2) LOAD TIME SERIES DATA
# =============================================================================
def load_time_series_vtk(
    file_start, file_end, file_step,
    vtk_path, vtk_field,
    X_scale=1.0, Y_scale=1.0,
    dt=0.02,
    threshold=-np.inf,
    verbose=True
):
    if file_step <= 0:
        raise ValueError("file_step must be >= 1")
    vtk_path_use = vtk_path if vtk_path.endswith("/") else (vtk_path + "/")
    file_indices = list(range(file_start, file_end + 1, file_step))
    if len(file_indices) == 0:
        raise RuntimeError("No file indices generated. Check start/end/step.")
    T_scale_time  = (file_end - file_start) * dt
    t_phys_start  = file_start * dt
    phys_times    = np.array([n * dt for n in file_indices], dtype=np.float64)
    norm_times    = (phys_times - t_phys_start) / T_scale_time
    if verbose:
        print(f"[TIME] dt            = {dt} s")
        print(f"[TIME] t_phys_start  = {t_phys_start:.4f} s  (file_start={file_start} × dt)")
        print(f"[TIME] T_scale_time  = {T_scale_time:.4f} s  "
              f"((file_end - file_start) = {file_end - file_start} × dt)")
        print(f"[TIME] t* range      : [{norm_times[0]:.4f}, {norm_times[-1]:.4f}]")
        print(f"[TIME] Physical range: [{phys_times[0]:.4f}, {phys_times[-1]:.4f}] s")
    X_total, Y_total, temp_total, t_total = [], [], [], []
    file_indices_loaded = []
    for idx, n in enumerate(file_indices):
        mesh_file = f"{vtk_path_use}Data_{n}.vtk"
        if verbose:
            print(f"Reading {mesh_file}")
        if not os.path.exists(mesh_file):
            if verbose:
                print(f"WARNING: Missing file {mesh_file} (skipping)")
            continue
        try:
            data_vtk = load_vtk_file(mesh_file)
            coords   = extract_coordinates(data_vtk)
            temp     = extract_field_array(data_vtk, vtk_field)
        except Exception as e:
            print(f"Error loading {mesh_file}: {e}")
            continue
        x      = coords[:, 0:1]
        y      = coords[:, 1:2]
        t_vals = np.full_like(x, float(norm_times[idx]))
        temp   = temp.reshape(-1, 1)
        X_total.append(x);    Y_total.append(y)
        temp_total.append(temp); t_total.append(t_vals)
        file_indices_loaded.append(n)
    if len(X_total) == 0:
        raise RuntimeError("No VTK files were successfully loaded.")
    x_all = np.vstack(X_total) / X_scale
    y_all = np.vstack(Y_total) / Y_scale
    T_all = np.vstack(temp_total)
    T_all = np.where(T_all < threshold, threshold, T_all)
    t_all = np.vstack(t_total)
    if verbose:
        for name, arr in [("X_all (norm)", x_all), ("Y_all (norm)", y_all),
                          ("T_all", T_all), ("Time_all (t*)", t_all)]:
            print(f"{name:>22}: min={np.min(arr):.6f}, max={np.max(arr):.6f}")
        print(f"[INFO] Loaded file indices: {file_indices_loaded}")
    return t_all, x_all, y_all, T_all, file_indices_loaded, T_scale_time
# =============================================================================
# 3) NETWORK — NEURON-WISE ADAPTIVE ACTIVATION
# =============================================================================
class AdaptiveSwishLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.a      = nn.Parameter(torch.ones(out_features))
    def forward(self, x):
        z = self.linear(x)
        return z * torch.sigmoid(self.a * z)
class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_frequencies, std):
        super().__init__()
        self.b = nn.Parameter(
            torch.randn(num_frequencies, input_dim) * std,
            requires_grad=False
        )
    def forward(self, x):
        b      = self.b.to(x.device)
        x_proj = x @ b.T
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
class AdaptiveFourierNet(nn.Module):
    def __init__(self, input_dim, h_n, output_dim, num_layers,
                 num_frequencies, std):
        super().__init__()
        self.fourier  = FourierFeatures(input_dim, num_frequencies, std)
        fourier_dim   = input_dim + 2 * num_frequencies
        layers = [AdaptiveSwishLayer(fourier_dim, h_n)]
        for _ in range(num_layers):
            layers.append(AdaptiveSwishLayer(h_n, h_n))
        layers.append(nn.Linear(h_n, output_dim))
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(self.fourier(x))
def init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
def build_models(h_n, num_layers, num_frequencies, std, device):
    net_u = AdaptiveFourierNet(2, h_n, 1, num_layers, num_frequencies, std).to(device)
    net_v = AdaptiveFourierNet(2, h_n, 1, num_layers, num_frequencies, std).to(device)
    net_p = AdaptiveFourierNet(2, h_n, 1, num_layers, num_frequencies, std).to(device)
    net_T = AdaptiveFourierNet(3, h_n, 1, num_layers, num_frequencies, std).to(device)
    for net in (net_u, net_v, net_p, net_T):
        net.apply(init_normal)
    return net_u, net_v, net_p, net_T
# =============================================================================
# 4) TRAINING
# =============================================================================
def geo_train(
    device,
    t_data, x_data, y_data, T_data,
    t_phys_in, x_phys_in, y_phys_in,          # physics pts INSIDE mask
    t_phys_out, x_phys_out, y_phys_out,        # physics pts OUTSIDE mask (full domain)
    batchsize_data, batchsize_phys,
    learning_rate, epochs,
    Diff, miu,
    inv_Re, inv_Pe,
    beta_x, beta_y, beta_t,
    Lambda_data, Lambda_con,
    Lambda_x, Lambda_y, Lambda_adv, Lambda_bc,
    alpha,                                      # weight multiplier for phys_out losses
    Flag_schedule, decay_rate, step_epoch,
    wall_vtk_file,
    X_scale, Y_scale, T_scale, U_scale, P_scale,
    h_n, num_layers, num_frequencies, std,
    path_weights, path_loss_npy, path_plots,
    run_tag
):
    """
    Loss equation:
      total = Lambda_data * L_data
            + Lambda_bc   * L_bc
            + Lambda_con  * L_con_in   + Lambda_x * L_xmom_in
            + Lambda_y    * L_ymom_in  + Lambda_adv * L_adv_in
            + alpha * Lambda_con  * L_con_out
            + alpha * Lambda_x    * L_xmom_out
            + alpha * Lambda_y    * L_ymom_out
            + alpha * Lambda_adv  * L_adv_out

    alpha = 0  →  phys_out losses vanish entirely
    alpha = 1  →  phys_out losses have the same weights as phys_in losses
    """
    # ---- tensors ----
    t_d = torch.tensor(t_data, dtype=torch.float32, device=device)
    x_d = torch.tensor(x_data, dtype=torch.float32, device=device)
    y_d = torch.tensor(y_data, dtype=torch.float32, device=device)
    T_d = torch.tensor(T_data, dtype=torch.float32, device=device)

    t_pi = torch.tensor(t_phys_in,  dtype=torch.float32, device=device)
    x_pi = torch.tensor(x_phys_in,  dtype=torch.float32, device=device)
    y_pi = torch.tensor(y_phys_in,  dtype=torch.float32, device=device)

    t_po = torch.tensor(t_phys_out, dtype=torch.float32, device=device)
    x_po = torch.tensor(x_phys_out, dtype=torch.float32, device=device)
    y_po = torch.tensor(y_phys_out, dtype=torch.float32, device=device)

    data_loader    = DataLoader(TensorDataset(t_d,  x_d,  y_d,  T_d),
                                batch_size=batchsize_data,  shuffle=True,
                                num_workers=0, drop_last=False)
    phys_in_loader = DataLoader(TensorDataset(t_pi, x_pi, y_pi),
                                batch_size=batchsize_phys, shuffle=True,
                                num_workers=0, drop_last=True)
    phys_out_loader = DataLoader(TensorDataset(t_po, x_po, y_po),
                                 batch_size=batchsize_phys, shuffle=True,
                                 num_workers=0, drop_last=True)

    xw_all, yw_all = load_wall_points(wall_vtk_file, X_scale, Y_scale, device)
    n_wall = xw_all.shape[0]
    if n_wall < 10:
        raise RuntimeError(f"Wall.vtk seems too small ({n_wall} points).")
    print(f"[INFO] Loaded wall points : {n_wall}")
    print(f"[INFO] beta_t             = {beta_t:.6f}")
    print(f"[INFO] alpha (phys_out)   = {alpha}")

    net_u, net_v, net_p, net_T = build_models(h_n, num_layers, num_frequencies,
                                               std, device)
    optimizer_u = optim.AdamW(net_u.parameters(), lr=learning_rate,
                              betas=(0.9, 0.99), eps=1e-15)
    optimizer_v = optim.AdamW(net_v.parameters(), lr=learning_rate,
                              betas=(0.9, 0.99), eps=1e-15)
    optimizer_p = optim.AdamW(net_p.parameters(), lr=learning_rate,
                              betas=(0.9, 0.99), eps=1e-15)
    optimizer_T = optim.AdamW(net_T.parameters(), lr=learning_rate,
                              betas=(0.9, 0.99), eps=1e-15)

    if Flag_schedule:
        scheduler_u = torch.optim.lr_scheduler.StepLR(
            optimizer_u, step_size=step_epoch, gamma=decay_rate)
        scheduler_v = torch.optim.lr_scheduler.StepLR(
            optimizer_v, step_size=step_epoch, gamma=decay_rate)
        scheduler_p = torch.optim.lr_scheduler.StepLR(
            optimizer_p, step_size=step_epoch, gamma=decay_rate)
        scheduler_T = torch.optim.lr_scheduler.StepLR(
            optimizer_T, step_size=step_epoch, gamma=decay_rate)

    mse = nn.MSELoss()

    def criterion_phys(t_in, x_in, y_in):
        """Compute the 4 physics residuals for a given set of collocation points."""
        t_in.requires_grad_(True)
        x_in.requires_grad_(True)
        y_in.requires_grad_(True)
        xy  = torch.cat((x_in, y_in), dim=1)
        u   = net_u(xy).view(len(x_in), -1)
        v   = net_v(xy).view(len(x_in), -1)
        p   = net_p(xy).view(len(x_in), -1)
        txy = torch.cat((t_in, x_in, y_in), dim=1)
        T   = net_T(txy).view(len(x_in), -1)
        def g(out, inp):
            return torch.autograd.grad(
                out, inp, grad_outputs=torch.ones_like(out),
                create_graph=True)[0]
        u_x  = g(u,   x_in);  u_y  = g(u,   y_in)
        u_xx = g(u_x, x_in);  u_yy = g(u_y, y_in)
        v_x  = g(v,   x_in);  v_y  = g(v,   y_in)
        v_xx = g(v_x, x_in);  v_yy = g(v_y, y_in)
        p_x  = g(p,   x_in);  p_y  = g(p,   y_in)
        T_x  = g(T,   x_in);  T_y  = g(T,   y_in)
        T_xx = g(T_x, x_in);  T_yy = g(T_y, y_in)
        T_t  = g(T,   t_in)

        r_cont = beta_x*u_x + beta_y*v_y
        r_xmom = (u*(beta_x*u_x) + v*(beta_y*u_y)) \
                 + (beta_x*p_x) \
                 - inv_Re*(beta_x**2*u_xx + beta_y**2*u_yy)
        r_ymom = (u*(beta_x*v_x) + v*(beta_y*v_y)) \
                 + (beta_y*p_y) \
                 - inv_Re*(beta_x**2*v_xx + beta_y**2*v_yy)
        r_adv  = beta_t*T_t \
                 + (u*(beta_x*T_x) + v*(beta_y*T_y)) \
                 - inv_Pe*(beta_x**2*T_xx + beta_y**2*T_yy)

        return (mse(r_cont, torch.zeros_like(r_cont)),
                mse(r_xmom, torch.zeros_like(r_xmom)),
                mse(r_ymom, torch.zeros_like(r_ymom)),
                mse(r_adv,  torch.zeros_like(r_adv)))

    def loss_data_fn(t_in, x_in, y_in, T_in):
        if t_in.dim() == 1: t_in = t_in.unsqueeze(1)
        if x_in.dim() == 1: x_in = x_in.unsqueeze(1)
        if y_in.dim() == 1: y_in = y_in.unsqueeze(1)
        if T_in.dim() == 1: T_in = T_in.unsqueeze(1)
        txy   = torch.cat((t_in, x_in, y_in), dim=1)
        out_T = net_T(txy) * T_scale
        return mse(out_T, T_in)

    def loss_bc_noslip_all():
        xyw = torch.cat((xw_all, yw_all), dim=1)
        u_w = net_u(xyw)
        v_w = net_v(xyw)
        return mse(u_w, torch.zeros_like(u_w)) + mse(v_w, torch.zeros_like(v_w))

    # ---- loss history tensors (8 physics + data + bc) ----
    Loss_CON_IN   = torch.zeros(epochs, 1)
    Loss_X_MOM_IN = torch.zeros(epochs, 1)
    Loss_Y_MOM_IN = torch.zeros(epochs, 1)
    Loss_ADV_IN   = torch.zeros(epochs, 1)
    Loss_CON_OUT   = torch.zeros(epochs, 1)
    Loss_X_MOM_OUT = torch.zeros(epochs, 1)
    Loss_Y_MOM_OUT = torch.zeros(epochs, 1)
    Loss_ADV_OUT   = torch.zeros(epochs, 1)
    Loss_DATA  = torch.zeros(epochs, 1)
    Loss_BC    = torch.zeros(epochs, 1)

    tic         = time.time()
    phys_in_iter  = iter(phys_in_loader)
    phys_out_iter = iter(phys_out_loader)

    for epoch in range(epochs):
        (lci, lxi, lyi, lai,
         lco, lxo, lyo, lao,
         ld, lbc) = [0.0] * 10

        for (t_d_b, x_d_b, y_d_b, T_d_b) in data_loader:

            # --- fetch phys_in batch ---
            try:
                (t_pi_b, x_pi_b, y_pi_b) = next(phys_in_iter)
            except StopIteration:
                phys_in_iter = iter(phys_in_loader)
                (t_pi_b, x_pi_b, y_pi_b) = next(phys_in_iter)

            # --- fetch phys_out batch (only needed when alpha > 0) ---
            if alpha > 0.0:
                try:
                    (t_po_b, x_po_b, y_po_b) = next(phys_out_iter)
                except StopIteration:
                    phys_out_iter = iter(phys_out_loader)
                    (t_po_b, x_po_b, y_po_b) = next(phys_out_iter)

            optimizer_u.zero_grad()
            optimizer_v.zero_grad()
            optimizer_p.zero_grad()
            optimizer_T.zero_grad()

            # physics residuals — inside mask
            con_in, xm_in, ym_in, adv_in = criterion_phys(t_pi_b, x_pi_b, y_pi_b)

            batch_data = loss_data_fn(t_d_b, x_d_b, y_d_b, T_d_b)
            batch_bc   = loss_bc_noslip_all()

            total = (Lambda_data * batch_data
                   + Lambda_bc   * batch_bc
                   + Lambda_con  * con_in
                   + Lambda_x    * xm_in
                   + Lambda_y    * ym_in
                   + Lambda_adv  * adv_in)

            # physics residuals — outside mask (full domain), scaled by alpha
            if alpha > 0.0:
                con_out, xm_out, ym_out, adv_out = criterion_phys(t_po_b, x_po_b, y_po_b)
                total = total + (alpha * Lambda_con  * con_out
                               + alpha * Lambda_x    * xm_out
                               + alpha * Lambda_y    * ym_out
                               + alpha * Lambda_adv  * adv_out)
            else:
                # still track zeros so history arrays stay populated
                con_out = xm_out = ym_out = adv_out = torch.tensor(0.0, device=device)

            total.backward()
            optimizer_u.step()
            optimizer_v.step()
            optimizer_p.step()
            optimizer_T.step()

            lci += con_in.detach();  lxi += xm_in.detach()
            lyi += ym_in.detach();   lai += adv_in.detach()
            lco += con_out.detach(); lxo += xm_out.detach()
            lyo += ym_out.detach();  lao += adv_out.detach()
            ld  += batch_data.detach()
            lbc += batch_bc.detach()

        N = len(data_loader)
        Loss_CON_IN[epoch]   = lci / N;  Loss_X_MOM_IN[epoch] = lxi / N
        Loss_Y_MOM_IN[epoch] = lyi / N;  Loss_ADV_IN[epoch]   = lai / N
        Loss_CON_OUT[epoch]  = lco / N;  Loss_X_MOM_OUT[epoch] = lxo / N
        Loss_Y_MOM_OUT[epoch] = lyo / N; Loss_ADV_OUT[epoch]  = lao / N
        Loss_DATA[epoch]     = ld  / N
        Loss_BC[epoch]       = lbc / N

        print(
            f"[EPOCH {epoch}]"
            f"  ContIN:{Loss_CON_IN[epoch].item():.3e}"
            f"  xMomIN:{Loss_X_MOM_IN[epoch].item():.3e}"
            f"  yMomIN:{Loss_Y_MOM_IN[epoch].item():.3e}"
            f"  AdvIN:{Loss_ADV_IN[epoch].item():.3e}"
            f"  ContOUT:{Loss_CON_OUT[epoch].item():.3e}"
            f"  xMomOUT:{Loss_X_MOM_OUT[epoch].item():.3e}"
            f"  yMomOUT:{Loss_Y_MOM_OUT[epoch].item():.3e}"
            f"  AdvOUT:{Loss_ADV_OUT[epoch].item():.3e}"
            f"  Data:{Loss_DATA[epoch].item():.3e}"
            f"  BC:{Loss_BC[epoch].item():.3e}"
            f"  LR:{optimizer_u.param_groups[0]['lr']:.1e}"
        )

        is_first       = (epoch == 0)
        is_last        = (epoch == epochs - 1)
        is_decay_epoch = Flag_schedule and (epoch > 0) and (epoch % step_epoch == 0)
        if (epochs == 1) or is_first or is_decay_epoch or is_last:
            torch.save(net_T.state_dict(),
                       os.path.join(path_weights, f"T_ep{epoch}.pt"))
            torch.save(net_u.state_dict(),
                       os.path.join(path_weights, f"U_ep{epoch}.pt"))
            torch.save(net_v.state_dict(),
                       os.path.join(path_weights, f"V_ep{epoch}.pt"))
            torch.save(net_p.state_dict(),
                       os.path.join(path_weights, f"P_ep{epoch}.pt"))

        if Flag_schedule:
            scheduler_u.step()
            scheduler_v.step()
            scheduler_p.step()
            scheduler_T.step()

    print(f"Elapsed: {time.time()-tic:.1f}s")

    # ---- save loss arrays ----
    loss_arrays = {
        "Loss_CON_IN":    Loss_CON_IN,
        "Loss_X_MOM_IN":  Loss_X_MOM_IN,
        "Loss_Y_MOM_IN":  Loss_Y_MOM_IN,
        "Loss_ADV_IN":    Loss_ADV_IN,
        "Loss_CON_OUT":   Loss_CON_OUT,
        "Loss_X_MOM_OUT": Loss_X_MOM_OUT,
        "Loss_Y_MOM_OUT": Loss_Y_MOM_OUT,
        "Loss_ADV_OUT":   Loss_ADV_OUT,
        "Loss_DATA":      Loss_DATA,
        "Loss_BC":        Loss_BC,
    }
    for name, arr in loss_arrays.items():
        np.save(os.path.join(path_loss_npy, f"{name}.npy"), arr.cpu().numpy())
    print(f"Loss .npy files saved to: {path_loss_npy}")

    # ---- plot: two sub-figures (in vs out physics, plus data/bc) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for label, arr in [
        ("Cont IN",  Loss_CON_IN),
        ("xMom IN",  Loss_X_MOM_IN),
        ("yMom IN",  Loss_Y_MOM_IN),
        ("Adv IN",   Loss_ADV_IN),
        ("Data",     Loss_DATA),
        ("BC",       Loss_BC),
    ]:
        ax.plot(arr.cpu().numpy(), label=label)
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Phys-IN / Data / BC losses"); ax.legend(fontsize=7)

    ax = axes[1]
    for label, arr in [
        ("Cont OUT",  Loss_CON_OUT),
        ("xMom OUT",  Loss_X_MOM_OUT),
        ("yMom OUT",  Loss_Y_MOM_OUT),
        ("Adv OUT",   Loss_ADV_OUT),
    ]:
        ax.plot(arr.cpu().numpy(), label=label)
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Phys-OUT losses  (alpha={alpha})"); ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(path_plots, "Loss.png"), dpi=300)
    plt.close()
# =============================================================================
# 5) INFERENCE + VTK OUTPUT + WSS  (unchanged)
# =============================================================================
def create_vtk_and_wss(
    t_np, x_np, y_np,
    mesh_file, output_filename, device,
    std, num_frequencies,
    U_scale, T_scale, P_scale, h_n, num_layers,
    run_tag,
    Lambda_data, Lambda_con, Lambda_x, Lambda_y, Lambda_adv, Lambda_bc,
    epochs, batchsize_data, batchsize_phys, step,
    path_weights, miu, X_scale, Y_scale,
    vtk_path_for_csv, wss_out_dir
):
    t = torch.tensor(t_np, dtype=torch.float32, device=device, requires_grad=True)
    x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
    y = torch.tensor(y_np, dtype=torch.float32, device=device, requires_grad=True)
    net_u, net_v, net_p, net_T = build_models(h_n, num_layers, num_frequencies,
                                               std, device)
    wT = os.path.join(path_weights, f"T_ep{step}.pt")
    wU = os.path.join(path_weights, f"U_ep{step}.pt")
    wV = os.path.join(path_weights, f"V_ep{step}.pt")
    wP = os.path.join(path_weights, f"P_ep{step}.pt")
    for wf in [wT, wU, wV, wP]:
        if not os.path.exists(wf):
            raise FileNotFoundError(f"Missing weight file: {wf}")
    net_T.load_state_dict(torch.load(wT, map_location=device))
    net_u.load_state_dict(torch.load(wU, map_location=device))
    net_v.load_state_dict(torch.load(wV, map_location=device))
    net_p.load_state_dict(torch.load(wP, map_location=device))
    for net in (net_u, net_v, net_p, net_T):
        net.eval()
    xy  = torch.cat((x, y), dim=1)
    u   = net_u(xy).view(-1, 1)
    v   = net_v(xy).view(-1, 1)
    p   = net_p(xy).view(-1, 1)
    txy = torch.cat((t, x, y), dim=1)
    T   = net_T(txy).view(-1, 1)
    u_np_out = u.detach().cpu().numpy() * U_scale
    v_np_out = v.detach().cpu().numpy() * U_scale
    p_np_out = p.detach().cpu().numpy() * P_scale
    T_np_out = T.detach().cpu().numpy() * T_scale
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points  = data_vtk.GetNumberOfPoints()
    if u_np_out.shape[0] != n_points:
        raise RuntimeError(
            f"Point count mismatch: model={u_np_out.shape[0]} vs VTK={n_points}.")
    for arr_np, name in [(np.column_stack([u_np_out[:,0], v_np_out[:,0]]), "Vel_PINN"),
                          (T_np_out, "Temp_PINN"),
                          (p_np_out, "Pressure_PINN")]:
        vtk_arr = VN.numpy_to_vtk(arr_np)
        vtk_arr.SetName(name)
        data_vtk.GetPointData().AddArray(vtk_arr)
    writer = vtk.vtkDataSetWriter()
    writer.SetInputData(data_vtk)
    writer.SetFileName(output_filename)
    writer.Write()
    # --- WSS ---
    print("\n===== WSS COMPUTATION START =====")
    du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                create_graph=False, retain_graph=False)[0]
    wss_pred  = miu * du_dy * (U_scale / Y_scale)
    x_phys_t  = x * X_scale
    wall_tol  = 1e-4
    pred_mask = (y.abs() < wall_tol) & (x_phys_t >= 1.0) & (x_phys_t <= 4.5)
    x_pred        = x_phys_t[pred_mask].detach().cpu().numpy().flatten()
    wss_pred_wall = wss_pred[pred_mask].detach().cpu().numpy().flatten()
    print(f"[INFO] Pred wall points: {len(x_pred)}")
    if len(x_pred) < 5:
        print("WARNING: Very few wall points detected.")
    idx_s         = np.argsort(x_pred)
    x_pred        = x_pred[idx_s]
    wss_pred_wall = wss_pred_wall[idx_s]
    csv_path = os.path.join(vtk_path_for_csv, "WSS.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing GT CSV: {csv_path}")
    gt_data = np.genfromtxt(csv_path, delimiter=",", names=True)
    x_gt    = gt_data["x"];  wss_gt = gt_data["WSS"]
    gt_mask = (x_gt >= 1.0) & (x_gt <= 4.5)
    x_gt = x_gt[gt_mask];  wss_gt = wss_gt[gt_mask]
    idx_g = np.argsort(x_gt)
    x_gt = x_gt[idx_g];  wss_gt = wss_gt[idx_g]
    x_common   = np.linspace(1.0, 4.5, 500)
    wss_pred_i = (np.interp(x_common, x_pred, wss_pred_wall)
                  if len(x_pred) > 1 else np.full_like(x_common, np.nan))
    wss_gt_i   = (np.interp(x_common, x_gt, wss_gt)
                  if len(x_gt) > 1 else np.full_like(x_common, np.nan))
    gt_norm = np.linalg.norm(wss_gt_i[np.isfinite(wss_gt_i)])
    diff    = wss_pred_i - wss_gt_i
    rel_l2  = (np.nan if gt_norm < 1e-12
               else np.linalg.norm(diff[np.isfinite(diff)]) / gt_norm)
    print(f"[RESULT] Relative L2 Error (WSS): {rel_l2}")
    ensure_dir(wss_out_dir)
    np.save(os.path.join(wss_out_dir, "x_pred.npy"),   x_pred)
    np.save(os.path.join(wss_out_dir, "WSS_pred.npy"), wss_pred_wall)
    np.save(os.path.join(wss_out_dir, "x_gt.npy"),     x_gt)
    np.save(os.path.join(wss_out_dir, "WSS_gt.npy"),   wss_gt)
    with open(os.path.join(wss_out_dir, "rel_l2.txt"), "w") as f:
        f.write(f"Relative L2 Error (WSS): {rel_l2}\n")
    for fig_fn, title, y_label, pairs in [
        ("WSS_vs_x.png", "Wall Shear Stress vs x", "WSS",
         [(x_gt, wss_gt, "GT WSS"), (x_pred, wss_pred_wall, "PINN WSS")]),
        ("WSS_error.png", "WSS Absolute Error vs x", "|WSS Error|",
         [(x_common, np.abs(wss_pred_i - wss_gt_i), "")]),
    ]:
        plt.figure()
        for xd, yd, lbl in pairs:
            plt.plot(xd, yd, linewidth=2, label=lbl) if lbl else plt.plot(xd, yd)
        plt.xlabel("x"); plt.ylabel(y_label); plt.title(title)
        if any(p[2] for p in pairs): plt.legend()
        plt.savefig(os.path.join(wss_out_dir, fig_fn), dpi=300)
        plt.close()
    print("[INFO] WSS results saved\n===== WSS COMPUTATION END =====\n")
def run_inference(
    file_start, file_end, file_step,
    vtk_input_path, path_results,
    X_scale, Y_scale,
    std, num_frequencies,
    U_scale, T_scale, P_scale, h_n, num_layers,
    run_tag,
    Lambda_data, Lambda_con, Lambda_x, Lambda_y, Lambda_adv, Lambda_bc,
    epochs, batchsize_data, batchsize_phys, step,
    path_weights, miu, wss_out_dir, device,
    dt=0.02,
    verbose=True
):
    vtk_input_path_use = (vtk_input_path if vtk_input_path.endswith("/")
                          else vtk_input_path + "/")
    file_indices = list(range(file_start, file_end + 1, file_step))
    if len(file_indices) == 0:
        raise RuntimeError("No inference indices generated.")
    T_scale_time = (file_end - file_start) * dt
    t_phys_start = file_start * dt
    phys_times   = np.array([n * dt for n in file_indices], dtype=np.float64)
    norm_times   = (phys_times - t_phys_start) / T_scale_time
    if verbose:
        print("Starting inference phase...")
    for idx, m in enumerate(file_indices):
        filename = f"{vtk_input_path_use}Data_{m}.vtk"
        if not os.path.exists(filename):
            print(f"WARNING: {filename} not found. Skipping.")
            continue
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(filename); reader.Update()
        data_vtk = reader.GetOutput()
        n_points  = data_vtk.GetNumberOfPoints()
        x_vtk = np.array([data_vtk.GetPoint(i)[0] for i in range(n_points)]).reshape(-1,1)
        y_vtk = np.array([data_vtk.GetPoint(i)[1] for i in range(n_points)]).reshape(-1,1)
        x = x_vtk / X_scale
        y = y_vtk / Y_scale
        t = np.full_like(x, float(norm_times[idx]))
        output_filename = os.path.join(path_results, f"PINN_{m}.vtk")
        create_vtk_and_wss(
            t_np=t, x_np=x, y_np=y,
            mesh_file=filename, output_filename=output_filename,
            device=device, std=std, num_frequencies=num_frequencies,
            U_scale=U_scale, T_scale=T_scale, P_scale=P_scale,
            h_n=h_n, num_layers=num_layers, run_tag=run_tag,
            Lambda_data=Lambda_data, Lambda_con=Lambda_con,
            Lambda_x=Lambda_x, Lambda_y=Lambda_y,
            Lambda_adv=Lambda_adv, Lambda_bc=Lambda_bc,
            epochs=epochs, batchsize_data=batchsize_data,
            batchsize_phys=batchsize_phys, step=step,
            path_weights=path_weights, miu=miu,
            X_scale=X_scale, Y_scale=Y_scale,
            vtk_path_for_csv=vtk_input_path_use.rstrip("/"),
            wss_out_dir=wss_out_dir
        )
        if verbose:
            print(f"Done: {filename}")
    if verbose:
        print("=" * 60)
        print("INFERENCE PHASE COMPLETED!")
        print("=" * 60)
# =============================================================================
# 6) MAIN
# =============================================================================
if __name__ == "__main__":
    script_name   = os.path.basename(sys.argv[0])
    match         = re.search(r'_(\d+)\.py$', script_name)
    experiment_id = int(match.group(1)) if match else -1
    print(f"[INFO] Experiment ID: {experiment_id}")
    script_start_time = time.time()
    device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_infer = torch.device("cpu")
    # ------------------------------------------------------------------ paths
    vtk_path      = "Results_BFS"
    vtk_field     = "C"
    threshold     = 0.0
    wall_vtk_file = os.path.join(vtk_path, "Wall.vtk")
    # ------------------------------------------------------------ y-OBSERVATION WINDOW
    # The PINN is only fed concentration data from one or two horizontal bands
    # in physical y (units of the mesh). Points outside these bands are dropped
    # by apply_two_region_y_mask() before training.
    #   Region 1 : keep data with  y1_min <= y <= y1_max          (always active)
    #   Region 2 : keep data with  y2_min <= y <= y2_max          (optional)
    #              set y2_min = y2_max = None to disable Region 2
    # NOTE: limits are in PHYSICAL y (the mask divides the network y by Y_scale
    #       internally). Keep this window consistent with the one used in the
    #       inverse solver (Diff_Physics.py).
    y1_min, y1_max = 0.0, 0.1       # Region 1 [y_min, y_max]
    y2_min, y2_max = None, None     # Region 2 [y_min, y_max]  (None = disabled)
    # ------------------------------------------------------------------ physics
    miu  = 0.04
    rho  = 1.06
    Diff = 0.001
    file_start = 0
    file_end   = 100
    file_step  = 5
    dt         = 0.02
    # ------------------------------------------------------------------ scales
    P_scale = 1.0
    T_scale = 1.0
    U_scale = 4.0
    L_scale = 0.5
    X_scale = 4.5
    Y_scale = 0.1
    beta_x = L_scale / X_scale
    beta_y = L_scale / Y_scale
    Re     = U_scale * L_scale * rho / miu
    Pe     = U_scale * L_scale / Diff
    inv_Re = 1.0 / Re
    inv_Pe = 1.0 / Pe
    T_scale_time = (file_end - file_start) * dt
    beta_t       = L_scale / (U_scale * T_scale_time)
    print(f"[INFO] T_scale_time = {T_scale_time:.4f} s")
    print(f"[INFO] beta_t       = {beta_t:.6f}")
    print(f"[INFO] beta_x       = {beta_x:.6f}")
    print(f"[INFO] beta_y       = {beta_y:.6f}")
    # ------------------------------------------------------------------ network
    h_n             = 100
    num_layers      = 8
    std             = 1.0
    num_frequencies = 10
    # ------------------------------------------------------------------ training
    data_sample_frac = 0.05
    phys_sample_frac = 0.25   # fraction of m → physics pts INSIDE the mask
    phys_out_frac    = 1.0   # fraction of m → physics pts from FULL domain
    batchsize_data   = 1024
    batchsize_phys   = 1024 * 8
    epochs           = 3001
    learning_rate    = 1e-3
    Flag_schedule    = True
    step_epoch       = max(1, int((epochs - 1) / 10)) if epochs > 1 else 1
    decay_rate       = 0.6
    step             = int(epochs - 1)
    # ------------------------------------------------------------------ alpha
    # alpha controls the weight of phys_out physics losses:
    #   0.0  →  phys_out losses are completely ignored
    #   1.0  →  phys_out losses receive the same Lambda weights as phys_in
    alpha = 0.2
    # ------------------------------------------------------------------ loss weights
    Lambda_data = 1e5
    Lambda_con  = 200
    Lambda_x    = 200
    Lambda_y    = 200
    Lambda_adv  = 200.0
    Lambda_bc   = 1000.0
    # ------------------------------------------------------------------ run folder
    runs_root      = ensure_dir("Runs")
    yregion_prefix = make_yregion_prefix(y1_min, y1_max, y2_min, y2_max)
    run_root, run_time = make_time_run_dir(os.path.join(runs_root, yregion_prefix))
    dirs = build_run_dirs(run_root)
    path_weights  = dirs["weights"]
    path_loss_npy = dirs["losses"]
    path_results  = dirs["results"]
    path_plots    = dirs["plots"]
    wss_out_dir   = dirs["wss"]
    print(f"[INFO] Run directory: {run_root}")
    # ------------------------------------------------------------------ run tag
    case_tag = (
        f"start{file_start}_end{file_end}_step{file_step}_Re{Re}_Pe{Pe}_"
        f"y1_{fmt_float_tag(y1_min)}_{fmt_float_tag(y1_max)}_"
        f"y2_{fmt_float_tag(y2_min)}_{fmt_float_tag(y2_max)}_"
        f"df{fmt_float_tag(data_sample_frac)}_pf{fmt_float_tag(phys_sample_frac)}_"
        f"alpha{fmt_float_tag(alpha)}"
    )
    model_tag = (
        f"hn{h_n}_l{num_layers}_f{num_frequencies}_std{fmt_float_tag(std)}_"
        f"lr{fmt_float_tag(learning_rate)}_ep{epochs}_"
        f"bsD{batchsize_data}_bsP{batchsize_phys}_"
        f"dec{fmt_float_tag(decay_rate)}_se{step_epoch}_"
        f"ld{fmt_float_tag(Lambda_data)}_lc{fmt_float_tag(Lambda_con)}_"
        f"lx{fmt_float_tag(Lambda_x)}_ly{fmt_float_tag(Lambda_y)}_"
        f"la{fmt_float_tag(Lambda_adv)}_lbc{fmt_float_tag(Lambda_bc)}_"
        f"Us{fmt_float_tag(U_scale)}_Ts{fmt_float_tag(T_scale)}_"
        f"Ps{fmt_float_tag(P_scale)}_Xs{fmt_float_tag(X_scale)}_"
        f"Ys{fmt_float_tag(Y_scale)}_mu{fmt_float_tag(miu)}"
    )
    run_tag = f"{case_tag}__{model_tag}"
    print("RUN TAG:", run_tag)
    # ------------------------------------------------------------------ metadata
    run_info = {
        "experiment_id":    experiment_id,
        "run_time":         run_time,
        "vtk_path":         vtk_path,
        "vtk_field":        vtk_field,
        "file_start":       file_start,
        "file_end":         file_end,
        "file_step":        file_step,
        "dt":               dt,
        "T_scale_time":     T_scale_time,
        "threshold":        threshold,
        "y1_min": y1_min, "y1_max": y1_max,
        "y2_min": y2_min, "y2_max": y2_max,
        "data_sample_frac": data_sample_frac,
        "phys_sample_frac": phys_sample_frac,
        "phys_out_frac":    phys_out_frac,
        "alpha":            alpha,
        "phys_sampling_note": (
            "phys_in  = int(phys_sample_frac * m) — drawn from INSIDE mask; "
            "phys_out = int(phys_out_frac    * m) — drawn from FULL domain; "
            "phys_out losses weighted by alpha * Lambda_X"
        ),
        "Re": Re, "Pe": Pe, "mu": miu, "Diff": Diff,
        "inv_Re": inv_Re, "inv_Pe": inv_Pe,
        "beta_x": beta_x, "beta_y": beta_y, "beta_t": beta_t,
        "X_scale": X_scale, "Y_scale": Y_scale,
        "U_scale": U_scale, "L_scale": L_scale,
        "T_scale": T_scale, "P_scale": P_scale,
        "hidden_neurons":   h_n,
        "num_layers":       num_layers,
        "num_frequencies":  num_frequencies,
        "std":              std,
        "epochs":           epochs,
        "learning_rate":    learning_rate,
        "batchsize_data":   batchsize_data,
        "batchsize_phys":   batchsize_phys,
        "scheduler": {
            "enabled":    Flag_schedule,
            "step_epoch": step_epoch,
            "decay_rate": decay_rate,
        },
        "Lambda_data": Lambda_data,
        "Lambda_con":  Lambda_con,
        "Lambda_x":    Lambda_x,
        "Lambda_y":    Lambda_y,
        "Lambda_adv":  Lambda_adv,
        "Lambda_bc":   Lambda_bc,
        "adaptive_activation": {
            "method": "neuron_wise_scale",
            "class":  "AdaptiveSwishLayer",
            "init_a": 1.0,
        },
    }
    # ------------------------------------------------------------------ load data
    t_all, x_all, y_all, T_all, _, T_scale_time_loaded = load_time_series_vtk(
        file_start=file_start, file_end=file_end, file_step=file_step,
        vtk_path=vtk_path, vtk_field=vtk_field,
        X_scale=X_scale, Y_scale=Y_scale,
        dt=dt, threshold=threshold, verbose=True
    )
    assert abs(T_scale_time_loaded - T_scale_time) < 1e-9, \
        f"T_scale_time mismatch: {T_scale_time_loaded} vs {T_scale_time}"
    # ------------------------------------------------------------------ mask & sample
    t_data, x_data, y_data, T_data = apply_two_region_y_mask(
        t=t_all, x=x_all, y=y_all, T=T_all, Y_scale=Y_scale,
        y1_min=y1_min, y1_max=y1_max,
        y2_min=y2_min, y2_max=y2_max, verbose=True
    )
    # --- data sampling (from masked region) ---
    n_d   = t_data.shape[0]
    t_data_full = t_data.copy()
    x_data_full = x_data.copy()
    y_data_full = y_data.copy()
    idx_d = np.random.choice(n_d, int(n_d * data_sample_frac), replace=False)
    t_data = t_data[idx_d]; x_data = x_data[idx_d]
    y_data = y_data[idx_d]; T_data = T_data[idx_d]
    # --- physics sampling ---
    m      = n_d
    n_full = t_all.shape[0]

    # (a) phys_in: INSIDE mask
    phys_in_count = int(phys_sample_frac * m)
    phys_in_count = min(phys_in_count, n_d)
    idx_pin    = np.random.choice(n_d, phys_in_count, replace=False)
    t_phys_in  = t_data_full[idx_pin].reshape(-1, 1)
    x_phys_in  = x_data_full[idx_pin].reshape(-1, 1)
    y_phys_in  = y_data_full[idx_pin].reshape(-1, 1)

    # (b) phys_out: FULL domain (NOT concatenated with phys_in)
    phys_out_count = int(phys_out_frac * m)
    phys_out_count = min(phys_out_count, n_full)
    idx_pout   = np.random.choice(n_full, phys_out_count, replace=False)
    t_phys_out = t_all[idx_pout].reshape(-1, 1)
    x_phys_out = x_all[idx_pout].reshape(-1, 1)
    y_phys_out = y_all[idx_pout].reshape(-1, 1)

    print(f"[INFO] Masked pts (m)              : {m}")
    print(f"[INFO] Data pts (sampled)          : {t_data.shape[0]}")
    print(f"[INFO] Physics pts inside mask     : {phys_in_count}  (phys_sample_frac={phys_sample_frac} × m)")
    print(f"[INFO] Physics pts full domain     : {phys_out_count} (phys_out_frac={phys_out_frac} × m)")
    print(f"[INFO] alpha (phys_out weight)     : {alpha}")

    run_info["m_masked_pts"]    = m
    run_info["phys_in_count"]   = phys_in_count
    run_info["phys_out_count"]  = phys_out_count
    run_info["n_full_pool"]     = n_full
    run_info["phys_sampling_note"] = (
        f"phys_in={phys_in_count} (={phys_sample_frac}*m inside mask); "
        f"phys_out={phys_out_count} (={phys_out_frac}*m from full domain); "
        f"phys_out losses scaled by alpha={alpha}"
    )
    with open(os.path.join(run_root, "RUN_INFO.json"), "w") as f:
        json.dump(run_info, f, indent=4)
    print("[INFO] RUN_INFO.json written")
    # ------------------------------------------------------------------ train
    geo_train(
        device=device_train,
        t_data=t_data, x_data=x_data, y_data=y_data, T_data=T_data,
        t_phys_in=t_phys_in, x_phys_in=x_phys_in, y_phys_in=y_phys_in,
        t_phys_out=t_phys_out, x_phys_out=x_phys_out, y_phys_out=y_phys_out,
        batchsize_data=batchsize_data, batchsize_phys=batchsize_phys,
        learning_rate=learning_rate, epochs=epochs,
        Diff=Diff, miu=miu,
        inv_Re=inv_Re, inv_Pe=inv_Pe,
        beta_x=beta_x, beta_y=beta_y, beta_t=beta_t,
        Lambda_data=Lambda_data,
        Lambda_con=Lambda_con, Lambda_x=Lambda_x,
        Lambda_y=Lambda_y, Lambda_adv=Lambda_adv, Lambda_bc=Lambda_bc,
        alpha=alpha,
        Flag_schedule=Flag_schedule, decay_rate=decay_rate, step_epoch=step_epoch,
        wall_vtk_file=wall_vtk_file,
        X_scale=X_scale, Y_scale=Y_scale, T_scale=T_scale,
        U_scale=U_scale, P_scale=P_scale,
        h_n=h_n, num_layers=num_layers,
        num_frequencies=num_frequencies, std=std,
        path_weights=path_weights, path_loss_npy=path_loss_npy,
        path_plots=path_plots,
        run_tag=run_tag
    )
    print(f"Total runtime: {time.time() - script_start_time:.1f}s")
