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
        for layer in list(self.net_psi) + list(self.net_E):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward_psi(self, x, V0, W):
        x_norm  = x  / 15.0
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0
        inp_pos = torch.cat([ x_norm, V0_norm, W_norm], dim=1)
        inp_neg = torch.cat([-x_norm, V0_norm, W_norm], dim=1)
        raw = (self.net_psi(inp_pos) + self.net_psi(inp_neg)) / 2.0
        # FIX 2: softplus → exp. Trivial sabit çözümü engeller.
        return torch.exp(raw)

    def forward_E(self, V0, W):
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0
        raw_E = self.net_E(torch.cat([V0_norm, W_norm], dim=1))
        # FIX 3: önce birim aralıkta öğren, sonra V0 ile ölçekle.
        E_frac = -torch.sigmoid(raw_E)   # ∈ (-1, 0)
        return E_frac * V0               # ∈ (-V0, 0)


def smooth_potential(x_col, V0_t, w_val, sharpness=80.0):
    """
    FIX 5: torch.where yerine differentiable smooth-step potansiyel.
    Kuyu içinde -V0, dışında 0'a yaklaşır; autograd ile tam uyumlu.
    """
    half_w = w_val / 2.0
    inside = torch.sigmoid(sharpness * (half_w - torch.abs(x_col)))
    return -V0_t * inside


# --- 2. EĞİTİM HAZIRLIĞI ---
L_domain   = 15.0
N_far      = 200   # Uzak bölge nokta sayısı
N_near     = 400   # Kuyu çevresi nokta sayısı (yoğunlaştırılmış)
BATCH_SIZE = 4
EPOCHS     = 25000

W_PDE  = 1.0
W_BC   = 100.0
W_NORM = 500.0

model     = SurrogatePINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

print("=" * 65)
print("Surrogate PINN Eğitimi Başlıyor (v3 — tüm kök nedenler düzeltildi)")
print(f"  Domain: [-{L_domain}, {L_domain}]  |  Colloc: {N_far+N_near}  |  Batch: {BATCH_SIZE}")
print(f"  W_PDE={W_PDE}  W_BC={W_BC}  W_NORM={W_NORM}")
print("=" * 65)

history = {"epoch": [], "total": [], "pde": [], "bc": [], "norm": []}

# --- 3. EĞİTİM DÖNGÜSÜ ---
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # FIX 1: torch.tensor(0.0) yerine loss listesi — computational graph kopmaz
    losses = []
    batch_pde = batch_bc = batch_norm = 0.0

    for _ in range(BATCH_SIZE):
        v0_val = float(np.random.uniform(2.0, 50.0))
        w_val  = float(np.random.uniform(1.0, 8.0))
        half_w = w_val / 2.0

        # FIX 4: Kuyu sınırı çevresinde yoğun örnekleme
        x_far  = torch.FloatTensor(N_far).uniform_(-L_domain, L_domain)
        x_near = torch.FloatTensor(N_near).uniform_(-half_w - 1.5, half_w + 1.5)
        x_col  = torch.cat([x_far, x_near]).unsqueeze(1).requires_grad_(True)
        N_tot  = x_col.shape[0]

        V0_t = torch.full((N_tot, 1), v0_val)
        W_t  = torch.full((N_tot, 1), w_val)

        psi    = model.forward_psi(x_col, V0_t, W_t)
        E_pred = model.forward_E(
            torch.tensor([[v0_val]]),
            torch.tensor([[w_val]])
        )

        dpsi_dx = torch.autograd.grad(
            psi, x_col,
            grad_outputs=torch.ones_like(psi),
            create_graph=True
        )[0]
        d2psi_dx2 = torch.autograd.grad(
            dpsi_dx, x_col,
            grad_outputs=torch.ones_like(dpsi_dx),
            create_graph=True
        )[0]

        # FIX 5: Yumuşatılmış potansiyel
        V_x = smooth_potential(x_col, V0_t, w_val)

        # Kayıp 1: PDE
        residual = -0.5 * d2psi_dx2 + V_x * psi - E_pred * psi
        loss_pde = W_PDE * torch.mean(residual ** 2)

        # Kayıp 2: Sınır koşulları ψ(±L) = 0
        x_bc   = torch.tensor([[-L_domain], [L_domain]])
        v0_bc  = torch.full((2, 1), v0_val)
        w_bc   = torch.full((2, 1), w_val)
        psi_bc = model.forward_psi(x_bc, v0_bc, w_bc)
        loss_bc = W_BC * torch.mean(psi_bc ** 2)

        # Kayıp 3: Normalizasyon ∫|ψ|² dx = 1
        # Düzgün grid üzerinde hesapla (trapz için)
        x_grid  = torch.linspace(-L_domain, L_domain, 800).view(-1, 1)
        V0_grid = torch.full((800, 1), v0_val)
        W_grid  = torch.full((800, 1), w_val)
        psi_g   = model.forward_psi(x_grid, V0_grid, W_grid)
        dx_g    = 2.0 * L_domain / 800
        norm_val  = torch.sum(psi_g ** 2) * dx_g
        loss_norm = W_NORM * (norm_val - 1.0) ** 2

        step_loss = loss_pde + loss_bc + loss_norm
        losses.append(step_loss)

        batch_pde  += loss_pde.item()
        batch_bc   += loss_bc.item()
        batch_norm += loss_norm.item()

    # FIX 1: stack ile ortalama — tüm graph bağlı
    total_loss = torch.stack(losses).mean()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    history["epoch"].append(epoch)
    history["total"].append(total_loss.item())
    history["pde"].append(batch_pde  / BATCH_SIZE)
    history["bc"].append(batch_bc   / BATCH_SIZE)
    history["norm"].append(batch_norm / BATCH_SIZE)

    if epoch % 1000 == 0:
        with torch.no_grad():
            e_s = model.forward_E(
                torch.tensor([[v0_val]]),
                torch.tensor([[w_val]])
            ).item()
        # Sonsuz kuyu alt sınırı referans (sonlu kuyu her zaman daha derin)
        e_inf_ref = -((np.pi ** 2) / (2.0 * w_val ** 2))
        print(
            f"Epoch {epoch:5d} | V0={v0_val:5.1f} W={w_val:4.1f} | "
            f"Loss={total_loss.item():.5f} | "
            f"E_PINN={e_s:.3f}  [sonsuz_ref≈{e_inf_ref:.3f}]"
        )

# --- 4. KAYDET ---
torch.save(model.state_dict(), "kuantum_beyni.pth")
print("\nEğitim Bitti! Model 'kuantum_beyni.pth' dosyasına kaydedildi.")

# --- 5. LOSS GRAFİĞİ ---
epochs_arr = history["epoch"]
fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.semilogy(epochs_arr, history["total"], color="royalblue", linewidth=1.5, label="Toplam Loss")
ax1.set_title("Toplam Eğitim Kaybı", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (log ölçek)")
ax1.legend(); ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.semilogy(epochs_arr, history["pde"],  color="tomato",          linewidth=1.5, label=f"PDE  (×{W_PDE})")
ax2.semilogy(epochs_arr, history["bc"],   color="darkorange",      linewidth=1.5, label=f"BC   (×{W_BC})")
ax2.semilogy(epochs_arr, history["norm"], color="mediumseagreen",  linewidth=1.5, label=f"Norm (×{W_NORM})")
ax2.set_title("Bileşen Kayıpları", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (log ölçek)")
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle("PINN Eğitim Geçmişi", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("egitim_loss.png", dpi=150, bbox_inches="tight")
plt.show()
print("Loss grafiği 'egitim_loss.png' olarak kaydedildi.")