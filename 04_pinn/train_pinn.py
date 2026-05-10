import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARAMETRİK PINN MİMARİSİ ---
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

        for layer in self.net_psi:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(layer.bias)

        for layer in self.net_E:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(layer.bias)

    def forward_psi(self, x, V0, W):
        """
        Normalizasyon hem burada hem app_pinn.py'de aynı şekilde yapılmalı.
        x: [-10, 10] -> [-1, 1]
        V0: [1, 50]  -> [0.02, 1]
        W:  [1, 8]   -> [0.125, 1]
        """
        x_norm  = x  / 10.0
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0

        # Çift-simetri zorlaması: ψ(x) = ψ(-x) (temel hal çift fonksiyon)
        inputs_pos = torch.cat([ x_norm, V0_norm, W_norm], dim=1)
        inputs_neg = torch.cat([-x_norm, V0_norm, W_norm], dim=1)

        psi = (self.net_psi(inputs_pos) + self.net_psi(inputs_neg)) / 2.0
        # softplus ile ψ ≥ 0 garantisi (temel hal nod içermez)
        return torch.nn.functional.softplus(psi)

    def forward_E(self, V0, W):
        """
        Enerji -V0 ile 0 arasında olmalı.
        sigmoid yerine normalize edilmiş girdi ile doğrudan tahmini öğretiyoruz,
        sonra fiziksel aralığa ölçekliyoruz.
        """
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0
        raw_E   = self.net_E(torch.cat([V0_norm, W_norm], dim=1))
        # -V0 < E < 0 aralığına sıkıştır
        return -V0 * torch.sigmoid(raw_E)


# --- 2. EĞİTİM HAZIRLIĞI ---
L_domain   = 15.0          # ↑ Sonsuz kuyu için sınırı uzattık (tunneling bölgesi)
N_colloc   = 1000          # Collocation nokta sayısı
BATCH_SIZE = 6             # Her adımda kaç farklı (V0, W) göreceğiz
EPOCHS     = 20000

# Kayıp ağırlıkları — PDE öğrenimi öncelikli
W_PDE  = 1.0
W_BC   = 50.0              # Sınır koşulu (eskiden 100, PDE ile dengeli olsun)
W_NORM = 200.0             # Normalizasyon (eskiden 1000, çok baskılıyordu)

model     = SurrogatePINN()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# Öğrenme hızını yavaşça düşür — son epoch'larda ince ayar
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

print("=" * 65)
print("Surrogate PINN Eğitimi Başlıyor...")
print(f"  Domain: [-{L_domain}, {L_domain}]  |  Colloc: {N_colloc}  |  Batch: {BATCH_SIZE}")
print(f"  W_PDE={W_PDE}  W_BC={W_BC}  W_NORM={W_NORM}")
print("=" * 65)

# Loss geçmişi
history = {"epoch": [], "total": [], "pde": [], "bc": [], "norm": []}

# --- 3. EĞİTİM DÖNGÜSÜ ---
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    total_batch_loss = torch.tensor(0.0)
    batch_pde = batch_bc = batch_norm = 0.0

    for _ in range(BATCH_SIZE):
        v0_val = float(np.random.uniform(1.0, 50.0))
        w_val  = float(np.random.uniform(1.0, 8.0))

        # Her mini-batch için TAZE collocation noktaları — gradyan kirlenmesi yok
        x_col = torch.linspace(-L_domain, L_domain, N_colloc).view(-1, 1)
        x_col.requires_grad_(True)
        dx = 2.0 * L_domain / N_colloc

        V0_t = torch.full_like(x_col, v0_val)
        W_t  = torch.full_like(x_col, w_val)

        # İleri geçiş
        psi    = model.forward_psi(x_col, V0_t, W_t)
        E_pred = model.forward_E(
            torch.tensor([[v0_val]]),
            torch.tensor([[w_val]])
        )

        # Schrödinger denklemi: -0.5 * ψ'' + V(x)*ψ = E*ψ  →  artık = 0
        dpsi_dx  = torch.autograd.grad(
            psi, x_col,
            grad_outputs=torch.ones_like(psi),
            create_graph=True
        )[0]
        d2psi_dx2 = torch.autograd.grad(
            dpsi_dx, x_col,
            grad_outputs=torch.ones_like(dpsi_dx),
            create_graph=True
        )[0]

        # Potansiyel: kuyu içinde -V0, dışında 0
        V_x = torch.where(
            torch.abs(x_col) < w_val / 2.0,
            -V0_t,
            torch.zeros_like(x_col)
        )

        # --- Kayıp Terimleri ---
        # 1) PDE artığı
        residual  = -0.5 * d2psi_dx2 + V_x * psi - E_pred * psi
        loss_pde  = W_PDE * torch.mean(residual ** 2)

        # 2) Sınır koşulları: ψ(±L) = 0
        x_bc  = torch.tensor([[-L_domain], [L_domain]])
        v0_bc = torch.tensor([[v0_val], [v0_val]])
        w_bc  = torch.tensor([[w_val],  [w_val]])
        psi_bc = model.forward_psi(x_bc, v0_bc, w_bc)
        loss_bc   = W_BC * torch.mean(psi_bc ** 2)

        # 3) Normalizasyon: ∫|ψ|² dx = 1
        norm_val  = torch.sum(psi ** 2) * dx
        loss_norm = W_NORM * (norm_val - 1.0) ** 2

        total_batch_loss = total_batch_loss + loss_pde + loss_bc + loss_norm
        batch_pde  += loss_pde.item()
        batch_bc   += loss_bc.item()
        batch_norm += loss_norm.item()

    total_loss = total_batch_loss / BATCH_SIZE
    total_loss.backward()

    # Gradyan patlamasını önle
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    # Her epoch'un loss değerini kaydet
    history["epoch"].append(epoch)
    history["total"].append(total_loss.item())
    history["pde"].append(batch_pde  / BATCH_SIZE)
    history["bc"].append(batch_bc   / BATCH_SIZE)
    history["norm"].append(batch_norm / BATCH_SIZE)

    if epoch % 1000 == 0:
        with torch.no_grad():
            e_sample = model.forward_E(
                torch.tensor([[v0_val]]),
                torch.tensor([[w_val]])
            ).item()
        print(
            f"Epoch {epoch:5d} | "
            f"V0={v0_val:5.1f} W={w_val:4.1f} | "
            f"Loss={total_loss.item():.5f} | "
            f"E_tahmin={e_sample:.3f} eV"
        )

# --- 4. MODELİ KAYDETME ---
torch.save(model.state_dict(), "kuantum_beyni.pth")
print("\nEğitim Bitti! Model 'kuantum_beyni.pth' dosyasına kaydedildi.")

# --- 5. LOSS GRAFİĞİ ---
epochs_arr = history["epoch"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sol: Toplam loss (log ölçeği)
ax1 = axes[0]
ax1.semilogy(epochs_arr, history["total"], color="royalblue", linewidth=1.5, label="Toplam Loss")
ax1.set_title("Toplam Eğitim Kaybı", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (log ölçek)")
ax1.legend()
ax1.grid(alpha=0.3)

# Sağ: Bileşen lossları ayrı ayrı (log ölçeği)
ax2 = axes[1]
ax2.semilogy(epochs_arr, history["pde"],  color="tomato",      linewidth=1.5, label=f"PDE  (×{W_PDE})")
ax2.semilogy(epochs_arr, history["bc"],   color="darkorange",  linewidth=1.5, label=f"BC   (×{W_BC})")
ax2.semilogy(epochs_arr, history["norm"], color="mediumseagreen", linewidth=1.5, label=f"Norm (×{W_NORM})")
ax2.set_title("Bileşen Kayıpları", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (log ölçek)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle("PINN Eğitim Geçmişi", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("egitim_loss.png", dpi=150, bbox_inches="tight")
plt.show()
print("Loss grafiği 'egitim_loss.png' olarak kaydedildi.")