import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Model mimarisi (train_pinn.py ile AYNI olmalı) ---
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
        # ÖNEMLİ: Normalizasyon train_pinn.py ile AYNI olmalı
        x_norm  = x  / 10.0
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0

        inputs_pos = torch.cat([ x_norm, V0_norm, W_norm], dim=1)
        inputs_neg = torch.cat([-x_norm, V0_norm, W_norm], dim=1)

        psi = (self.net_psi(inputs_pos) + self.net_psi(inputs_neg)) / 2.0
        return torch.nn.functional.softplus(psi)

    def forward_E(self, V0, W):
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0
        raw_E   = self.net_E(torch.cat([V0_norm, W_norm], dim=1))
        return -V0 * torch.sigmoid(raw_E)


print("=" * 60)
print("HIZLI KUANTUM ÇÖZÜCÜYE HOŞ GELDİNİZ (Yapay Zeka Destekli)")
print("=" * 60)

# --- Model Yükleme ---
model = SurrogatePINN()
try:
    model.load_state_dict(torch.load("kuantum_beyni.pth", weights_only=True))
    model.eval()
    print("Model başarıyla yüklendi.\n")
except FileNotFoundError:
    print("HATA: 'kuantum_beyni.pth' bulunamadı.")
    print("Önce 'train_pinn.py' dosyasını çalıştırın!")
    sys.exit(1)

# --- Kullanıcı Girdisi ---
while True:
    try:
        user_v0 = float(input("Kuyu Derinliğini girin (1 ile 50 arası): "))
        user_w  = float(input("Kuyu Genişliğini girin (1 ile 8 arası): "))
        if 1 <= user_v0 <= 50 and 1 <= user_w <= 8:
            break
        print("Lütfen belirtilen aralıklarda değer girin.")
    except ValueError:
        print("Lütfen sayısal bir değer girin.")

# --- Tahmin ---
L_domain = 15.0  # train_pinn.py ile aynı domain
x_plot   = torch.linspace(-L_domain, L_domain, 1000).view(-1, 1)
V0_t     = torch.full_like(x_plot, user_v0)
W_t      = torch.full_like(x_plot, user_w)

with torch.no_grad():
    psi_plot = model.forward_psi(x_plot, V0_t, W_t).numpy().flatten()
    E_final  = model.forward_E(
        torch.tensor([[user_v0]]),
        torch.tensor([[user_w]])
    ).item()

print(f"\nTahmin Edilen Temel Hal Enerjisi: E = {E_final:.4f} eV")

# --- Analitik Karşılaştırma (referans) ---
# Sonsuz kuyu için analitik değer: E_1 = π²/(2*W²)  (ℏ=m=1 birimleri)
E_analytic_inf = (np.pi ** 2) / (2.0 * user_w ** 2)
print(f"Sonsuz Kuyu Analitik Referansı:   E = {E_analytic_inf:.4f} eV")
print(f"(Not: Sonlu kuyu için |E| < V0={user_v0} ve E_sonlu < E_sonsuz beklenir)\n")

# --- Görselleştirme ---
x_np  = x_plot.numpy().flatten()
V_plot = np.where(np.abs(x_np) < user_w / 2.0, -user_v0, 0.0)

# Dalga fonksiyonunu enerji ekseninde görsel konuma taşı
max_psi = np.max(np.abs(psi_plot))
scale   = (user_v0 * 0.4) / max_psi if max_psi > 1e-8 else 1.0
psi_visual = psi_plot * scale + E_final

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sol: Potansiyel + Dalga Fonksiyonu
ax1 = axes[0]
ax1.fill_between(x_np, V_plot, -user_v0 * 1.3, color='steelblue', alpha=0.15, label='Kuyu Bölgesi')
ax1.plot(x_np, V_plot, color='black', linewidth=2, label=f'V(x)  [V0={user_v0}, W={user_w}]')
ax1.axhline(E_final, color='red', linestyle='--', linewidth=2,
            label=f'E (PINN) = {E_final:.3f} eV')
ax1.plot(x_np, psi_visual, color='royalblue', linewidth=2,
         label='ψ₀(x)  (ölçekli)')
ax1.set_title("Potansiyel ve Dalga Fonksiyonu", fontsize=13, fontweight='bold')
ax1.set_xlabel("Konum (x)")
ax1.set_ylabel("Enerji (eV)")
ax1.set_ylim(-user_v0 * 1.3, user_v0 * 0.3)
ax1.set_xlim(-L_domain, L_domain)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Sağ: Sadece |ψ|² (olasılık yoğunluğu)
ax2 = axes[1]
prob = psi_plot ** 2
prob /= np.trapz(prob, x_np)   # normalize et
ax2.plot(x_np, prob, color='darkorange', linewidth=2, label='|ψ₀(x)|²')
ax2.fill_between(x_np, prob, alpha=0.2, color='orange')
ax2.axvline(-user_w / 2, color='gray', linestyle=':', linewidth=1.5, label='Kuyu duvarları')
ax2.axvline( user_w / 2, color='gray', linestyle=':', linewidth=1.5)
ax2.set_title("Olasılık Yoğunluğu  |ψ₀(x)|²", fontsize=13, fontweight='bold')
ax2.set_xlabel("Konum (x)")
ax2.set_ylabel("|ψ|²")
ax2.set_xlim(-L_domain, L_domain)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.suptitle(
    f"PINN Sonucu  |  V0={user_v0} eV,  W={user_w}  |  E = {E_final:.3f} eV",
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.show()
print("Grafik 'kuantum_sonuc.png' olarak kaydedildi.")