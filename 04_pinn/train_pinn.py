import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. MİMARİ ---
class SurrogatePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_psi = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.net_E = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward_psi(self, x, V0, W):
        x_norm  = x  / 15.0
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0
        
        inp_pos = torch.cat([ x_norm, V0_norm, W_norm], dim=1)
        inp_neg = torch.cat([-x_norm, V0_norm, W_norm], dim=1)
        
        # Fiziksel Simetri (Çift Fonksiyon - Ground State)
        raw = (self.net_psi(inp_pos) + self.net_psi(inp_neg)) / 2.0
        
        # GÜNCELLEME: Hard Boundary Condition (Ansatz)
        # x = ±15 olduğunda envelope=0 olur. Bu sayede BC Loss'a gerek kalmaz.
        envelope = 1.0 - (x_norm ** 2)
        return envelope * raw 

    def forward_E(self, V0, W):
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0
        raw_E = self.net_E(torch.cat([V0_norm, W_norm], dim=1))
        E_frac = -torch.sigmoid(raw_E) 
        return E_frac * V0


def smooth_potential(x_col, V0_t, w_val, sharpness=80.0):
    half_w = w_val / 2.0
    inside = torch.sigmoid(sharpness * (half_w - torch.abs(x_col)))
    return -V0_t * inside


# --- 2. EĞİTİM HAZIRLIĞI ---
L_domain   = 15.0
N_far      = 200
N_near     = 400
BATCH_SIZE = 8 # Batch size'ı biraz artırmak stabiliteyi artırır
EPOCHS     = 25000

W_PDE  = 1.0
W_NORM = 500.0
W_VAR  = 20.0  # Varyasyonel Enerji Kaybı Ağırlığı

model     = SurrogatePINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

history = {"epoch": [], "total": [], "pde": [], "norm": [], "var": []}

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    losses = []
    
    for _ in range(BATCH_SIZE):
        v0_val = float(np.random.uniform(2.0, 50.0))
        w_val  = float(np.random.uniform(1.0, 8.0))
        half_w = w_val / 2.0

        x_far  = torch.FloatTensor(N_far).uniform_(-L_domain, L_domain)
        x_near = torch.FloatTensor(N_near).uniform_(-half_w - 1.5, half_w + 1.5)
        x_col  = torch.cat([x_far, x_near]).unsqueeze(1).requires_grad_(True)
        
        V0_t = torch.full((x_col.shape[0], 1), v0_val)
        W_t  = torch.full((x_col.shape[0], 1), w_val)

        psi    = model.forward_psi(x_col, V0_t, W_t)
        E_pred = model.forward_E(torch.tensor([[v0_val]]), torch.tensor([[w_val]]))

        dpsi_dx = torch.autograd.grad(psi, x_col, torch.ones_like(psi), create_graph=True)[0]
        d2psi_dx2 = torch.autograd.grad(dpsi_dx, x_col, torch.ones_like(dpsi_dx), create_graph=True)[0]

        V_x = smooth_potential(x_col, V0_t, w_val)

        # 1. PDE Kaybı
        residual = -0.5 * d2psi_dx2 + V_x * psi - E_pred * psi
        loss_pde = W_PDE * torch.mean(residual ** 2)

        # 2. Normalizasyon Kaybı
        x_grid  = torch.linspace(-L_domain, L_domain, 800).view(-1, 1)
        psi_g   = model.forward_psi(x_grid, torch.full((800,1), v0_val), torch.full((800,1), w_val))
        dx_g    = 2.0 * L_domain / 800
        norm_val  = torch.sum(psi_g ** 2) * dx_g
        loss_norm = W_NORM * (norm_val - 1.0) ** 2

        # 3. YENİ: Varyasyonel Enerji Kaybı (E'yi minimize etmeye zorlar)
        # <E> = integral( 0.5|psi'|^2 + V|psi|^2 )
        energy_density = 0.5 * (dpsi_dx ** 2) + V_x * (psi ** 2)
        loss_var = W_VAR * torch.mean(energy_density)

        losses.append(loss_pde + loss_norm + loss_var)

    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {total_loss.item():.6f} | E_pred: {E_pred.item():.3f}")

torch.save(model.state_dict(), "kuantum_beyni.pth")