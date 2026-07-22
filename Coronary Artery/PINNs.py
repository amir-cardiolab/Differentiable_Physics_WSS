import os
import torch
import numpy as np
#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import vtk
from vtk.util import numpy_support as VN
import random
torch.cuda.empty_cache()
path_weights = 'WEIGHTS_pre/'
path_Loss = 'Loss_pre/'
path_results = 'Results_pre/'
for path in [path_weights, path_Loss, path_results]:
    os.makedirs(path, exist_ok=True)
device = torch.device("cuda")
file_start = 175
file_end = 179
Flag_batch = True
Flag_schedule = True
# SOAP (Improving and Stabilizing Shampoo using Adam) is a second-order optimizer
# that works better with larger batch sizes. From: https://github.com/nikhilvyas/SOAP
# Your fixed default values
Lambda_bc = 100
Lambda_data = 200000
Lambda_eq = 10
Lambda_con = 25
Lambda_x = Lambda_eq
Lambda_z = Lambda_eq
Lambda_y = Lambda_eq
Lambda_adv = 10
# ------------------------------------------------------------------
# alpha: weight multiplier for physics losses on the FULL domain
#   alpha = 0  ->  only physics inside the data domain is penalised
#   alpha > 0  ->  additionally penalise physics on Full_Domain.vtk
#                  (losses weighted by alpha * Lambda_X)
# ------------------------------------------------------------------
alpha = 0.0
h_n = 70
num_layers = 8
batchsize = 1024 * 6
epochs = 6001
if (Flag_schedule):
        learning_rate = 1e-3
        step_epoch = int((epochs-1)/6)
        decay_rate = 0.5
print(f"Step epoch: {step_epoch}")
print(f"Alpha     : {alpha}")
step = int(epochs-1)
miu = 0.04
rho = 1.06
Diff = 0.005
dyn_vis = miu/rho
Z_scale = 0.35
X_scale = 0.28
Y_scale = 1.1
P_scale = 1
T_scale = 0.9
U_scale = 225
std = 0.1
num_frequencies = 10
L_scale = 0.9
Re = U_scale*L_scale/dyn_vis
Pe = L_scale*U_scale/Diff
inv_Re = 1/Re
inv_Pe = 1/Pe
def geo_train(device, xb, yb, zb, t, xd, yd, zd, Td, xfd, yfd, zfd, tfd, batchsize, learning_rate, epochs, path, Diff, rho, Lambda_BC,Lambda_data, alpha, Flag_schedule):
# def geo_train(device,xd, yd, zd, Td, batchsize, learning_rate, epochs, path, Diff, rho, Lambda_BC,Lambda_data, Flag_schedule):
    if (Flag_batch):

        xb = torch.Tensor(xb).to(device)
        yb = torch.Tensor(yb).to(device)
        zb = torch.Tensor(zb).to(device)

        t = torch.Tensor(t).to(device)
        xd = torch.Tensor(xd).to(device)
        yd = torch.Tensor(yd).to(device)
        zd = torch.Tensor(zd).to(device)

        Td = torch.Tensor(Td).to(device)

        # full-domain physics tensors (alpha only controls their loss weight)
        xfd = torch.Tensor(xfd).to(device)
        yfd = torch.Tensor(yfd).to(device)
        zfd = torch.Tensor(zfd).to(device)
        tfd = torch.Tensor(tfd).to(device)

        if(1):
            xb = xb.type(torch.cuda.FloatTensor)
            yb = yb.type(torch.cuda.FloatTensor)
            zb = zb.type(torch.cuda.FloatTensor)
            t = t.type(torch.cuda.FloatTensor)
            xd = xd.type(torch.cuda.FloatTensor)
            yd = yd.type(torch.cuda.FloatTensor)
            zd = zd.type(torch.cuda.FloatTensor)

            Td = Td.type(torch.cuda.FloatTensor)

            xfd = xfd.type(torch.cuda.FloatTensor)
            yfd = yfd.type(torch.cuda.FloatTensor)
            zfd = zfd.type(torch.cuda.FloatTensor)
            tfd = tfd.type(torch.cuda.FloatTensor)

        dataset = TensorDataset(t, xd, yd, zd, Td)
        dataloader = DataLoader(dataset, batch_size = batchsize, shuffle = True, num_workers = 0, drop_last = True)

        # full-domain physics loader (alpha scales its contribution to the loss)
        fd_dataset = TensorDataset(tfd, xfd, yfd, zfd)
        fd_dataloader = DataLoader(fd_dataset, batch_size = batchsize, shuffle = True, num_workers = 0, drop_last = True)


        input_n = 3
        input_s = 4

    class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace
        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)
    class FourierFeatures(nn.Module):
        def __init__(self, input_dim, num_frequencies, std):
            super(FourierFeatures, self).__init__()
            # Generate `b` values from a normal distribution with std deviation `std`
            self.b = nn.Parameter(torch.randn(num_frequencies, input_dim) * std, requires_grad=False)  # Store as a non-trainable parameter
        def forward(self, x):
            # Move `self.b` to the same device as `x`
            b = self.b.to(x.device)
            # Compute sin(b * x) and cos(b * x)
            x_proj = x @ b.T  # Shape: (batch_size, num_frequencies)
            sin_features = torch.sin(x_proj)
            cos_features = torch.cos(x_proj)
            # Concatenate original inputs, sin, and cos features
            return torch.cat([x, sin_features, cos_features], dim=-1)
    class SingleScaleFourierNet(nn.Module):
        def __init__(self, input_dim, h_n, output_dim, num_layers, num_frequencies, std):
            super(SingleScaleFourierNet, self).__init__()
            # Fourier feature extractor
            self.fourier = FourierFeatures(input_dim, num_frequencies, std)

            # Adjusted input size: original inputs + Fourier features
            fourier_dim = input_dim + 2 * num_frequencies

            # Main network
            layers = [nn.Linear(fourier_dim, h_n), Swish()]
            for _ in range(num_layers):
                layers.append(nn.Linear(h_n, h_n))
                layers.append(Swish())
            layers.append(nn.Linear(h_n, output_dim))

            self.main = nn.Sequential(*layers)
        def forward(self, x):
            # Apply Fourier features
            x = self.fourier(x)
            # Pass through the main network
            return self.main(x)
    net2_u = SingleScaleFourierNet(input_n, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    net2_v = SingleScaleFourierNet(input_n, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    net2_w = SingleScaleFourierNet(input_n, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    net2_p = SingleScaleFourierNet(input_n, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    net2_T = SingleScaleFourierNet(input_s, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
    net2_u.apply(init_normal)
    net2_v.apply(init_normal)
    net2_w.apply(init_normal)
    net2_p.apply(init_normal)
    net2_T.apply(init_normal)
    print("Using AdamW optimizer")
    optimizer_u = optim.AdamW(net2_u.parameters(), lr = learning_rate, betas = (0.9, 0.99), eps = 10**-15)
    optimizer_v = optim.AdamW(net2_v.parameters(), lr = learning_rate, betas = (0.9, 0.99), eps = 10**-15)
    optimizer_w = optim.AdamW(net2_w.parameters(), lr = learning_rate, betas = (0.9, 0.99), eps = 10**-15)
    optimizer_p = optim.AdamW(net2_p.parameters(), lr = learning_rate, betas = (0.9, 0.99), eps = 10**-15)
    optimizer_T = optim.AdamW(net2_T.parameters(), lr = learning_rate, betas = (0.9, 0.99), eps = 10**-15)

    def criterion(t,x,y,z):
        t.requires_grad = True
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        net_in = torch.cat((x,y,z), 1)
        u = net2_u(net_in)
        u = u.view(len(u), -1)
        v = net2_v(net_in)
        v = v.view(len(v), -1)
        w = net2_w(net_in)
        w = w.view(len(w), -1)
        P = net2_p(net_in)
        P = P.view(len(P), -1)
        net_in2 = torch.cat((t,x,y,z), 1)
        T = net2_T(net_in2)
        T = T.view(len(T), -1)

        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        u_z = torch.autograd.grad(u,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
        u_zz = torch.autograd.grad(u_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
        v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        v_z = torch.autograd.grad(v,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
        v_zz = torch.autograd.grad(v_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
        w_x = torch.autograd.grad(w,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        w_xx = torch.autograd.grad(w_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        w_y = torch.autograd.grad(w,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        w_yy = torch.autograd.grad(w_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        w_z = torch.autograd.grad(w,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
        w_zz = torch.autograd.grad(w_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
        P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        P_z = torch.autograd.grad(P,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]

        T_x = torch.autograd.grad(T,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        T_xx = torch.autograd.grad(T_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        T_y = torch.autograd.grad(T,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        T_yy = torch.autograd.grad(T_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        T_z = torch.autograd.grad(T,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
        T_zz = torch.autograd.grad(T_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
        T_t = torch.autograd.grad(T,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
        # Calculate losses for continuity, momentum, and energy equations
        loss_1 = u_x + v_y + w_z
        loss_2 = (u*u_x + v*u_y + w*u_z) + P_x - (inv_Re*(u_xx+u_yy+u_zz))
        loss_3 = (u*v_x + v*v_y + w*v_z) + P_y - (inv_Re*(v_xx+v_yy+v_zz))
        loss_4 = (u*w_x + v*w_y + w*w_z) + P_z - (inv_Re*(w_xx+w_yy+w_zz))
        loss_5 = T_t + (u*T_x + v*T_y + w*T_z) - (inv_Pe * (T_xx+T_yy+T_zz))
        loss_f = nn.MSELoss()

        loss_continuty  = loss_f(loss_1,torch.zeros_like(loss_1))
        loss_x_momentum = loss_f(loss_2,torch.zeros_like(loss_2))
        loss_y_momentum = loss_f(loss_3,torch.zeros_like(loss_3))
        loss_z_momentum = loss_f(loss_4,torch.zeros_like(loss_4))
        loss_adv_diff   = loss_f(loss_5,torch.zeros_like(loss_5))

        return loss_continuty, loss_x_momentum, loss_y_momentum, loss_z_momentum, loss_adv_diff


    def Loss_BC2(xb, yb, zb):
        net_in = torch.cat((xb,yb,zb), 1)
        out1_u = net2_u(net_in)
        out1_u = out1_u.view(len(out1_u), -1)
        out1_v = net2_v(net_in)
        out1_v = out1_v.view(len(out1_v), -1)
        out1_w = net2_w(net_in)
        out1_w = out1_w.view(len(out1_w), -1)


        loss_f = nn.MSELoss()

        loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v)) + loss_f(out1_w, torch.zeros_like(out1_w))
        return loss_noslip

    def Loss_data(t,xd,yd,zd,Td):
        net_in1 = torch.cat((t,xd,yd,zd), 1)
        out1_T = net2_T(net_in1)
        out1_T = out1_T.view(len(out1_T), -1) * T_scale

        loss_f = nn.MSELoss()
        loss_d = loss_f(out1_T, Td)

        return loss_d
    tic = time.time()
    if (Flag_schedule):

        scheduler_T_NN = torch.optim.lr_scheduler.StepLR(optimizer_T,  step_size=step_epoch, gamma=decay_rate)
        scheduler_U_NN = torch.optim.lr_scheduler.StepLR(optimizer_u,  step_size=step_epoch, gamma=decay_rate)
        scheduler_V_NN = torch.optim.lr_scheduler.StepLR(optimizer_v,  step_size=step_epoch, gamma=decay_rate)
        scheduler_W_NN = torch.optim.lr_scheduler.StepLR(optimizer_w,  step_size=step_epoch, gamma=decay_rate)
        scheduler_P_NN = torch.optim.lr_scheduler.StepLR(optimizer_p,  step_size=step_epoch, gamma=decay_rate)
    # Initialize loss history tensors for each component
    Loss_CON    = torch.zeros(epochs, 1)
    Loss_X_MOM  = torch.zeros(epochs, 1)
    Loss_Y_MOM  = torch.zeros(epochs, 1)
    Loss_Z_MOM  = torch.zeros(epochs, 1)
    Loss_ADV    = torch.zeros(epochs, 1)
    Loss_DATA   = torch.zeros(epochs, 1)
    Loss_BC     = torch.zeros(epochs, 1)
    # full-domain (OUT) physics loss history
    Loss_CON_OUT    = torch.zeros(epochs, 1)
    Loss_X_MOM_OUT  = torch.zeros(epochs, 1)
    Loss_Y_MOM_OUT  = torch.zeros(epochs, 1)
    Loss_Z_MOM_OUT  = torch.zeros(epochs, 1)
    Loss_ADV_OUT    = torch.zeros(epochs, 1)
# Access example: loss_trackers['CON'][epoch] = ...
    # iterator for the full-domain loader (always active; alpha scales the loss)
    fd_iter = iter(fd_dataloader)
    for epoch in range(epochs):
        loss_con = 0.0
        loss_x_mom = 0.0
        loss_y_mom = 0.0
        loss_z_mom = 0.0
        loss_adv = 0.0
        loss_data = 0.0
        loss_bc = 0.0
        loss_con_out = 0.0
        loss_x_mom_out = 0.0
        loss_y_mom_out = 0.0
        loss_z_mom_out = 0.0
        loss_adv_out = 0.0
        for batch_idx, (t_in1, x_in1, y_in1, z_in1, T_in1) in enumerate(dataloader):
            # Zero gradients for each optimizer
            optimizer_u.zero_grad()
            optimizer_v.zero_grad()
            optimizer_w.zero_grad()
            optimizer_p.zero_grad()
            optimizer_T.zero_grad()
            # Compute losses
            batch_con, batch_x_mom, batch_y_mom, batch_z_mom, batch_adv = criterion(t_in1, x_in1, y_in1, z_in1)
            batch_data = Loss_data(t_in1, x_in1, y_in1, z_in1, T_in1)
            batch_bc = Loss_BC2(xb, yb, zb)

            # --- physics on the FULL domain (Full_Domain.vtk); alpha scales its weight ---
            # Forward pass is ALWAYS computed so the OUT losses are always tracked.
            #   alpha = 0  ->  contribution is zero (no gradient from full domain)
            #   alpha > 0  ->  contribution = alpha * Lambda_X * loss_out
            try:
                (tfd_b, xfd_b, yfd_b, zfd_b) = next(fd_iter)
            except StopIteration:
                fd_iter = iter(fd_dataloader)
                (tfd_b, xfd_b, yfd_b, zfd_b) = next(fd_iter)
            batch_con_out, batch_x_mom_out, batch_y_mom_out, batch_z_mom_out, batch_adv_out = criterion(tfd_b, xfd_b, yfd_b, zfd_b)

            # Total loss
            loss = (Lambda_con * batch_con) + (Lambda_x * batch_x_mom)  + (Lambda_y * batch_y_mom) + (Lambda_z * batch_z_mom) + (Lambda_adv * batch_adv) + (Lambda_data * batch_data) + (Lambda_bc * batch_bc)
            loss = loss + (alpha * Lambda_con * batch_con_out) + (alpha * Lambda_x * batch_x_mom_out) + (alpha * Lambda_y * batch_y_mom_out) + (alpha * Lambda_z * batch_z_mom_out) + (alpha * Lambda_adv * batch_adv_out)
            loss.backward()
            # Optimizer steps
            with torch.no_grad():
                optimizer_u.step()
                optimizer_v.step()
                optimizer_w.step()
                optimizer_T.step()
                optimizer_p.step()
            # Update losses
            loss_con += batch_con.detach()
            loss_x_mom += batch_x_mom.detach()
            loss_y_mom += batch_y_mom.detach()
            loss_z_mom += batch_z_mom.detach()
            loss_adv += batch_adv.detach()
            loss_data += batch_data.detach()
            loss_bc += batch_bc.detach()
            # Update full-domain (OUT) losses
            loss_con_out += batch_con_out.detach()
            loss_x_mom_out += batch_x_mom_out.detach()
            loss_y_mom_out += batch_y_mom_out.detach()
            loss_z_mom_out += batch_z_mom_out.detach()
            loss_adv_out += batch_adv_out.detach()

        N = len(dataloader)  # Number of batches
        Loss_CON[epoch] = loss_con / N
        Loss_X_MOM[epoch] = loss_x_mom / N
        Loss_Y_MOM[epoch] = loss_y_mom / N
        Loss_Z_MOM[epoch] = loss_z_mom / N
        Loss_ADV[epoch] = loss_adv / N
        Loss_DATA[epoch] = loss_data / N
        Loss_BC[epoch] = loss_bc / N
        Loss_CON_OUT[epoch] = loss_con_out / N
        Loss_X_MOM_OUT[epoch] = loss_x_mom_out / N
        Loss_Y_MOM_OUT[epoch] = loss_y_mom_out / N
        Loss_Z_MOM_OUT[epoch] = loss_z_mom_out / N
        Loss_ADV_OUT[epoch] = loss_adv_out / N
        print(f'Total Loss [EPOCH = {epoch}]: \nContinuity Loss: {Loss_CON[epoch].item()} \nX-Momentum Loss: {Loss_X_MOM[epoch].item()} \nY-Momentum Loss: {Loss_Y_MOM[epoch].item()} \nZ-Momentum-Loss : {Loss_Z_MOM[epoch].item()} \nAdv-Diff-Loss: {Loss_ADV[epoch].item()} \n'
            f'Data Loss: {Loss_DATA[epoch].item()} \nBC Loss: {Loss_BC[epoch].item()}\n')
        print(f'[FULL-DOMAIN OUT | alpha={alpha}] CON_OUT: {Loss_CON_OUT[epoch].item()} X_MOM_OUT: {Loss_X_MOM_OUT[epoch].item()} Y_MOM_OUT: {Loss_Y_MOM_OUT[epoch].item()} Z_MOM_OUT: {Loss_Z_MOM_OUT[epoch].item()} ADV_OUT: {Loss_ADV_OUT[epoch].item()}\n')

        print(f"LEARNING RATE: {optimizer_u.param_groups[0]['lr']}")
        # Save model weights periodically
        if epoch % (epochs-1) == 0:
            # Add optimizer type to filename with short abbreviations
            opt = "A"  # S=SOAP, A=AdamW

            torch.save(net2_T.state_dict(), path_weights + f"T_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{epoch}.pt")
            torch.save(net2_u.state_dict(), path_weights + f"U_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{epoch}.pt")
            torch.save(net2_v.state_dict(), path_weights + f"V_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{epoch}.pt")
            torch.save(net2_w.state_dict(), path_weights + f"W_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{epoch}.pt")
            torch.save(net2_p.state_dict(), path_weights + f"P_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{epoch}.pt")
        # Learning rate scheduling
        if Flag_schedule:
            scheduler_T_NN.step()
            scheduler_P_NN.step()
            scheduler_U_NN.step()
            scheduler_V_NN.step()
            scheduler_W_NN.step()
    # Time tracking and loss plotting
    toc = time.time()
    elapseTime = toc - tic
    print("elapse time in parallel = ", elapseTime)
    plt.plot(Loss_CON, label='Continuity Los')
    plt.plot(Loss_X_MOM, label = 'X-Mom Loss')
    plt.plot(Loss_Y_MOM, label = 'Y-Mom Loss')
    plt.plot(Loss_Z_MOM, label = 'Z-Mom Loss')
    plt.plot(Loss_ADV, label = 'Adv Diff Loss')
    plt.plot(Loss_DATA, label='Data Loss')
    plt.plot(Loss_BC, label='BC Loss')
    plt.plot(Loss_CON_OUT, label='Continuity OUT')
    plt.plot(Loss_X_MOM_OUT, label='X-Mom OUT')
    plt.plot(Loss_Y_MOM_OUT, label='Y-Mom OUT')
    plt.plot(Loss_Z_MOM_OUT, label='Z-Mom OUT')
    plt.plot(Loss_ADV_OUT, label='Adv OUT')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Loss (alpha={alpha})')
    # Add optimizer type to loss figure filename with abbreviated names
    opt = "A"
    plt.savefig(f'Loss_pre/Loss_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}.jpg')
    # plt.show()

    return
### Main Code ###
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
# ------------------ Configuration ------------------
threshold = 0.0
vtk_field = 'Concentration'
vtk_path = "CFD_Data/"
wall_file = f"{vtk_path}Wall.vtk"
print(f'Files to be loaded: {file_start} to {file_end}')
# ----------------------------------------------------
# Number of timesteps
num_timesteps = file_end - file_start + 1
time_array = np.linspace(0, 1, num_timesteps)
X_total, Y_total, Z_total, temp_total, t_total = [], [], [], [], []
for idx, n in enumerate(range(file_start, file_end + 1)):
    mesh_file = f"{vtk_path}Data_{n}.vtk"
    print(f"Reading {mesh_file}")
    try:
        data_vtk = load_vtk_file(mesh_file)
        coords = extract_coordinates(data_vtk)
        temp = extract_field_array(data_vtk, vtk_field)
    except Exception as e:
        print(f"Error loading {mesh_file}: {e}")
        continue
    x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]

    # Scale time between 0 and 1 based on number of files
    t_val = time_array[idx]
    t_vals = np.full_like(x, t_val)
    temp = temp.reshape(-1, 1)
    X_total.append(x)
    Y_total.append(y)
    Z_total.append(z)
    temp_total.append(temp)
    t_total.append(t_vals)
# Convert and normalize data
X = np.vstack(X_total) / X_scale
Y = np.vstack(Y_total) / Y_scale
Z = np.vstack(Z_total) / Z_scale
T = np.vstack(temp_total)
T = np.where(T < threshold, threshold, T)
t = np.vstack(t_total)
# Process wall boundary mesh
print(f"\nLoading boundary file: {wall_file}")
wall_data = load_vtk_file(wall_file)
wall_coords = extract_coordinates(wall_data)
xb = wall_coords[:, 0:1] / X_scale
yb = wall_coords[:, 1:2] / Y_scale
zb = wall_coords[:, 2:3] / Z_scale
# Process full-domain physics points (Full_Domain.vtk)
# Always loaded; alpha only controls how much weight these losses get.
full_domain_file = f"{vtk_path}Full_Domain.vtk"
print(f"\nLoading full-domain file: {full_domain_file}")
fd_data = load_vtk_file(full_domain_file)
fd_coords = extract_coordinates(fd_data)
n_fd = fd_coords.shape[0]
# replicate across all timesteps (same spatial points, all time values)
xfd_list, yfd_list, zfd_list, tfd_list = [], [], [], []
for idx in range(num_timesteps):
    t_val = time_array[idx]
    xfd_list.append(fd_coords[:, 0:1] / X_scale)
    yfd_list.append(fd_coords[:, 1:2] / Y_scale)
    zfd_list.append(fd_coords[:, 2:3] / Z_scale)
    tfd_list.append(np.full((n_fd, 1), t_val))
xfd = np.vstack(xfd_list)
yfd = np.vstack(yfd_list)
zfd = np.vstack(zfd_list)
tfd = np.vstack(tfd_list)
print(f"Full-domain points loaded: {xfd.shape[0]} ({n_fd} spatial x {num_timesteps} timesteps)")
print(f"alpha = {alpha}  ->  full-domain loss contribution = alpha * Lambda_X * loss_out")
# Final reshape
xd = X
yd = Y
zd = Z
Td = T
# Summary output
def summarize_field(name, arr):
    print(f"{name:>20}: min = {np.min(arr):.4f}, max = {np.max(arr):.4f}")
print("\n===== Data Summary =====")
summarize_field("Xb input", xb)
summarize_field("Yb input", yb)
summarize_field("Zb input", zb)
summarize_field("Xd input", xd)
summarize_field("Yd input", yd)
summarize_field("Zd input", zd)
summarize_field("Td input", Td)
summarize_field("Time d input", t)
summarize_field("Xfd input", xfd)
summarize_field("Yfd input", yfd)
summarize_field("Zfd input", zfd)
# Run training
path = ""
geo_train(
    device, xb, yb, zb,
    t, xd, yd, zd, Td,
    xfd, yfd, zfd, tfd,
    batchsize, learning_rate, epochs, path,
    Diff, rho, Lambda_bc, Lambda_data, alpha, Flag_schedule
)
def compute_wss_and_save_vtk(input_filename, velocity_field_name, output_filename):
    # Read the input file
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(input_filename)
    reader.Update()
    # Get the input data
    data = reader.GetOutput()
    n_points= data.GetNumberOfPoints()
    # Compute the gradient of the velocity field
    gradientFilter = vtk.vtkGradientFilter()
    gradientFilter.SetInputData(data)
    gradientFilter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, velocity_field_name)
    gradientFilter.Update()
    # Extract the surface
    surfaceFilter = vtk.vtkDataSetSurfaceFilter()
    surfaceFilter.SetInputData(gradientFilter.GetOutput())
    surfaceFilter.Update()
    # Generate surface normals
    normalsFilter = vtk.vtkPolyDataNormals()
    normalsFilter.SetInputData(surfaceFilter.GetOutput())
    normalsFilter.Update()
    # Get the output data with normals
    normalsData = normalsFilter.GetOutput()
    # Get the velocity gradient and normals as numpy arrays
    gradient = VN.vtk_to_numpy(normalsData.GetPointData().GetArray("Gradients"))
    normals = VN.vtk_to_numpy(normalsData.GetPointData().GetNormals())
    # Compute the traction vector (sigma) components
    sigma_1 = 0.04 * (2 * gradient[:, 0] * normals[:, 0] +
                    (gradient[:, 1] + gradient[:, 3]) * normals[:, 1] +
                    (gradient[:, 2] + gradient[:, 6]) * normals[:, 2])
    sigma_2 = 0.04 * ((gradient[:, 1] + gradient[:, 3]) * normals[:, 0] +
                    2 * gradient[:, 4] * normals[:, 1] +
                    (gradient[:, 5] + gradient[:, 7]) * normals[:, 2])
    sigma_3 = 0.04 * ((gradient[:, 2] + gradient[:, 6]) * normals[:, 0] +
                    (gradient[:, 5] + gradient[:, 7]) * normals[:, 1] +
                    2 * gradient[:, 8] * normals[:, 2])
    # Compute the traction vector (tract_vector)
    tract_vector = np.vstack((sigma_1, sigma_2, sigma_3)).T
    # Compute the norm_vector
    norm_vector = normals
    # Compute the Wall Shear Stress (WSS)
    dot_product = np.einsum('ij,ij->i', tract_vector, norm_vector)
    norm_vector_scaled = norm_vector * dot_product[:, np.newaxis]
    wss = tract_vector - norm_vector_scaled
    # Convert WSS to VTK format and add to the output data
    wss_vtk = np.zeros((n_points,3))
    wss_vtk = wss[:,0]
    wss_vtk = VN.numpy_to_vtk(wss)
    wss_vtk.SetName('PINN_WSS')
    normalsData.GetPointData().AddArray(wss_vtk)
    # Write the output to a .vtk file
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(normalsData)
    writer.Write()
    print('Done Computing WSS')
def set_wall_velocity_to_zero(original_filename, wall_filename, output_filename):
    # Read the original .vtk file
    original_reader = vtk.vtkDataSetReader()
    original_reader.SetFileName(original_filename)
    original_reader.Update()
    original_dataset = original_reader.GetOutput()
    # Read the wall points .vtk file
    wall_reader = vtk.vtkDataSetReader()
    wall_reader.SetFileName(wall_filename)
    wall_reader.Update()
    wall_dataset = wall_reader.GetOutput()
    # Get the velocity data (assuming it's stored as point data and named 'Vel_PINN')
    velocity_array = original_dataset.GetPointData().GetArray('Vel_PINN')
    if velocity_array is None:
        raise ValueError("No 'Vel_PINN' array found in the original file.")
    # Create a new array to store the modified velocities
    new_velocity_array = vtk.vtkFloatArray()
    new_velocity_array.SetName('Vel_PINN')
    new_velocity_array.SetNumberOfComponents(velocity_array.GetNumberOfComponents())
    new_velocity_array.SetNumberOfTuples(velocity_array.GetNumberOfTuples())
    # Initialize new_velocity_array with original velocity data
    for i in range(velocity_array.GetNumberOfTuples()):
        new_velocity_array.SetTuple(i, velocity_array.GetTuple(i))
    # Iterate over the wall points and set the corresponding velocities to zero in the original dataset
    wall_points = wall_dataset.GetPoints()
    modified_points_count = 0
    for i in range(wall_points.GetNumberOfPoints()):
        wall_point = wall_points.GetPoint(i)
        point_id = original_dataset.FindPoint(wall_point)
        if point_id != -1:
            new_velocity_array.SetTuple(point_id, (0.0, 0.0, 0.0))
            modified_points_count += 1
    # Output the number of modified points
    print(f"Number of points with velocity set to zero: {modified_points_count}")
    # Update the original dataset with the new velocity data
    original_dataset.GetPointData().AddArray(new_velocity_array)
    # Write the modified data to the output .vtk file
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(original_dataset)
    writer.Write()
def create_vtk(t,x,y,z, mesh_file):
    print(f"Starting create_vtk for {mesh_file}")

    t = torch.Tensor(t).to(device)
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    z = torch.Tensor(z).to(device)

    print("Tensors created and moved to device")

    input_n = 4
    input_s = 3

    class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace
        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)
    class FourierFeatures(nn.Module):
        def __init__(self, input_dim, num_frequencies, std):
            super(FourierFeatures, self).__init__()
            # Generate b values from a normal distribution with std deviation std
            self.b = nn.Parameter(torch.randn(num_frequencies, input_dim) * std, requires_grad=False)  # Store as a non-trainable parameter
        def forward(self, x):
            # Move self.b to the same device as x
            b = self.b.to(x.device)
            # Compute sin(b * x) and cos(b * x)
            x_proj = x @ b.T  # Shape: (batch_size, num_frequencies)
            sin_features = torch.sin(x_proj)
            cos_features = torch.cos(x_proj)
            # Concatenate original inputs, sin, and cos features
            return torch.cat([x, sin_features, cos_features], dim=-1)
    class SingleScaleFourierNet(nn.Module):
        def __init__(self, input_dim, h_n, output_dim, num_layers, num_frequencies, std):
            super(SingleScaleFourierNet, self).__init__()
            # Fourier feature extractor
            self.fourier = FourierFeatures(input_dim, num_frequencies, std)

            # Adjusted input size: original inputs + Fourier features
            fourier_dim = input_dim + 2 * num_frequencies

            # Main network
            layers = [nn.Linear(fourier_dim, h_n), Swish()]
            for _ in range(num_layers):
                layers.append(nn.Linear(h_n, h_n))
                layers.append(Swish())
            layers.append(nn.Linear(h_n, output_dim))

            self.main = nn.Sequential(*layers)
        def forward(self, x):
            # Apply Fourier features
            x = self.fourier(x)
            # Pass through the main network
            return self.main(x)
    net2_u = SingleScaleFourierNet(input_s, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    net2_v = SingleScaleFourierNet(input_s, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    net2_w = SingleScaleFourierNet(input_s, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    net2_p = SingleScaleFourierNet(input_s, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    net2_T = SingleScaleFourierNet(input_n, h_n, 1, num_layers, num_frequencies, std=std).to(device)
    print("Loading the network weights...")
    print(f"Looking for weights at epoch {step}")

    # Check if weight files exist (use same optimizer type as training)
    opt = "A"  # S=SOAP, A=AdamW
    weight_files = [
        f"T_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{step}.pt",
        f"U_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{step}.pt",
        f"V_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{step}.pt",
        f"W_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{step}.pt",
        f"P_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{step}.pt"
    ]

    missing_files = []
    for wf in weight_files:
        if not os.path.exists(path_weights + wf):
            missing_files.append(wf)

    if missing_files:
        print(f"ERROR: Missing weight files: {missing_files}")
        print(f"Available files in {path_weights}:")
        if os.path.exists(path_weights):
            available = os.listdir(path_weights)
            print(available[:10])  # Show first 10 files
        else:
            print("Weight directory doesn't exist!")
        return

    print("All weight files found, loading...")

    try:
        net2_T.load_state_dict(torch.load(path_weights + weight_files[0]))
        print("Loaded T weights")
        net2_u.load_state_dict(torch.load(path_weights + weight_files[1]))
        print("Loaded U weights")
        net2_v.load_state_dict(torch.load(path_weights + weight_files[2]))
        print("Loaded V weights")
        net2_w.load_state_dict(torch.load(path_weights + weight_files[3]))
        print("Loaded W weights")
        net2_p.load_state_dict(torch.load(path_weights + weight_files[4]))
        print("Loaded P weights")
    except Exception as e:
        print(f"ERROR loading weights: {e}")
        return
    net2_u.eval()
    net2_v.eval()
    net2_w.eval()
    net2_p.eval()
    net2_T.eval()
    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    print("Starting forward pass...")

    net_in = torch.cat((x,y,z), 1)
    print(f"Created net_in with shape: {net_in.shape}")
    print("Computing velocity components...")
    u = net2_u((net_in))
    u = u.view(len(u), -1)
    print("U computed")

    v = net2_v((net_in))
    v = v.view(len(v), -1)
    print("V computed")

    w = net2_w((net_in))
    w = w.view(len(w), -1)
    print("W computed")

    p = net2_p((net_in))
    p = p.view(len(p), -1)
    p = p.data.numpy()
    p = p * P_scale
    print("P computed")

    net_in2 = torch.cat((t,x,y,z), 1)
    T = net2_T((net_in2))
    T = T.view(len(T), -1)
    T = T.data.numpy()
    T = T * T_scale
    u = u.data.numpy()
    v = v.data.numpy()
    w = w.data.numpy()
    u = u * U_scale
    v = v * U_scale
    w = w * U_scale
    mesh_file2 = mesh_file
    # Read the VTK dataset
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(mesh_file2)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    point_data = vtk.vtkPolyData()
    point_data.ShallowCopy(data_vtk)
    print(n_points)
    Velocity = np.zeros((n_points,3))
    Velocity[:,0] = u[:,0]
    Velocity[:,1] = v[:,0]
    Velocity[:,2] = w[:,0]
    theta_vtk = VN.numpy_to_vtk(Velocity)
    theta_vtk.SetName('Vel_PINN')
    data_vtk.GetPointData().AddArray(theta_vtk)
    myoutput = vtk.vtkDataSetWriter()
    myoutput.SetInputData(data_vtk)
    myoutput.SetFileName(output_filename)
    myoutput.Write()
    Temp = np.zeros((n_points,1))
    Temp[:,0] =  T[:,0]
    theta_vtk = VN.numpy_to_vtk(Temp)
    theta_vtk.SetName('Temp_PINN')
    data_vtk.GetPointData().AddArray(theta_vtk)
    myoutput = vtk.vtkDataSetWriter()

    myoutput.SetInputData(data_vtk)

    myoutput.SetFileName(output_filename)

    myoutput.Write()

    Pressure = np.zeros((n_points,1))
    Pressure[:,0] =  p[:,0]
    theta_vtk = VN.numpy_to_vtk(Pressure)
    theta_vtk.SetName('Pressure_PINN')
    data_vtk.GetPointData().AddArray(theta_vtk)
    myoutput = vtk.vtkDataSetWriter()

    myoutput.SetInputData(data_vtk)

    myoutput.SetFileName(output_filename)

    myoutput.Write()
    if Lambda_bc ==0:
        set_wall_velocity_to_zero(output_filename, "CFD_Data/Wall.vtk",output_filename)
    compute_wss_and_save_vtk(output_filename, 'Vel_PINN', output_filename_2)

    print('Done!')
#########
device = torch.device("cpu")
# Compute time values scaled from 0 to 1
num_files = file_end - file_start + 1
time_array = np.linspace(0, 1, num_files)
print("Starting inference phase...")
print("Looking for VTK files...")
# Check if CFD_Data directory exists
import os
if not os.path.exists("CFD_Data/"):
    print("ERROR: CFD_Data/ directory not found!")
    print("Available directories:", [d for d in os.listdir('.') if os.path.isdir(d)])
    print("Skipping inference phase...")
else:
    print("Found CFD_Data/ directory")
    vtk_files = [f for f in os.listdir("CFD_Data/") if f.endswith('.vtk')]
    print(f"Available VTK files: {vtk_files}")
for idx, m in enumerate(range(file_start, file_end + 1)):
    filename = f"CFD_Data/Data_{m}.vtk"
    print(f"Processing {filename}")

    # Check if file exists
    if not os.path.exists(filename):
        print(f"ERROR: File {filename} not found! Skipping...")
        continue

    try:
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(filename)
        reader.Update()
        data_vtk = reader.GetOutput()
        n_points = data_vtk.GetNumberOfPoints()
        print("Number of points:", n_points)
    except Exception as e:
        print(f"ERROR reading {filename}: {e}")
        continue
    # Temp Data probe (redundant reader removed)
    probe = vtk.vtkProbeFilter()
    point_data = vtk.vtkPolyData()
    point_data.ShallowCopy(data_vtk)
    probe.SetInputData(point_data)
    probe.SetSourceData(data_vtk)
    probe.Update()
    print('n_points of the mesh:', n_points)
    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        z_vtk_mesh[i] = pt_iso[2]
    # Normalized spatial coordinates
    x = np.reshape(x_vtk_mesh, (-1, 1)) / X_scale
    y = np.reshape(y_vtk_mesh, (-1, 1)) / Y_scale
    z = np.reshape(z_vtk_mesh, (-1, 1)) / Z_scale
    # Normalized time input using linspace
    t_val = time_array[idx]
    t = np.full_like(x, t_val)
    opt = "A"  # S=SOAP, A=AdamW
    output_filename = f"Results_pre/PINN_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{step}_{m}.vtk"
    print('Output file name:', output_filename)
    output_filename_2 = f"Results_pre/PINN_{opt}_s{std}_f{num_frequencies}_us{U_scale}_ld{Lambda_data}_lb{Lambda_bc}_lc{Lambda_con}_lx{Lambda_x}_ly{Lambda_y}_lz{Lambda_z}_la{Lambda_adv}_al{alpha}_e{epochs}_ps{P_scale}_b{batchsize}_n{h_n}_l{num_layers}_f1{file_start}_f2{file_end}_ep{step}_WSS_{m}.vtk"
    print('Time', t)
    try:
        create_vtk(t, x, y, z, filename)
        print(f"Successfully processed {filename}")
    except Exception as e:
        print(f"ERROR in create_vtk for {filename}: {e}")
print("="*60)
print("INFERENCE PHASE COMPLETED!")
print("="*60)
