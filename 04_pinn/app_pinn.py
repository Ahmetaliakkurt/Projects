import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Model (train_pinn.py ile BİREBİR AYNI olmalı) ---
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
        
        # Simetri (Çift Fonksiyon)
        raw = (self.net_psi(inp_pos) + self.net_psi(inp_neg)) / 2.0
        
        # Ansatz: x = ±15'te dalga fonksiyonunu kesinlikle sıfırlar
        envelope = 1.0 - (x_norm ** 2)
        return envelope * raw 

    def forward_E(self, V0, W):
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0
        raw_E   = self.net_E(torch.cat([V0_norm, W_norm], dim=1))
        E_frac  = -torch.sigmoid(raw_E)
        return E_frac * V0


print("=" * 60)
print("HIZLI KUANTUM ÇÖZÜCÜYE HOŞ GELDİNİZ (Yapay Zeka Destekli)")
print("=" * 60)

# --- Model Yükleme ---
model = SurrogatePINN()
try:
    # weights_only=True güvenli yükleme sağlar
    model.load_state_dict(torch.load("kuantum_beyni.pth", weights_only=True))
    model.eval()
    print("Model başarıyla yüklendi.\n")
except FileNotFoundError:
    print("HATA: 'kuantum_beyni.pth' bulunamadı.")
    print("Önce güncellenmiş 'train_pinn.py' dosyasını çalıştırın!")
    sys.exit(1)
except RuntimeError as e:
    print("HATA: Model ağırlıkları uyuşmuyor.")
    print("Eski 'kuantum_beyni.pth' dosyasını silin ve eğitimi yeniden başlatın!")
    sys.exit(1)

# --- Kullanıcı Girdisi ---
while True:
    try:
        user_v0 = float(input("Kuyu Derinliğini girin (2 ile 50 arası): "))
        user_w  = float(input("Kuyu Genişliğini girin (1 ile 8 arası): "))
        if 2 <= user_v0 <= 50 and 1 <= user_w <= 8:
            break
        print("Lütfen belirtilen aralıklarda değer girin.")
    except ValueError:
        print("Lütfen sayısal bir değer girin.")

# --- Tahmin (Inference) ---
L_domain = 15.0
x_plot   = torch.linspace(-L_domain, L_domain, 1000).view(-1, 1)
V0_t     = torch.full_like(x_plot, user_v0)
W_t      = torch.full_like(x_plot, user_w)

with torch.no_grad():
    psi_plot = model.forward_psi(x_plot, V0_t, W_t).numpy().flatten()
    E_final  = model.forward_E(
        torch.tensor([[user_v0]]),
        torch.tensor([[user_w]])
    ).item()

# Analitik referans (Sonsuz kuyu — sonlu kuyu için alt sınır)
E_analytic_inf = -((np.pi ** 2) / (2.0 * user_w ** 2))

print(f"\nTahmin Edilen Temel Hal Enerjisi : E = {E_final:.4f} eV")
print(f"Sonsuz Kuyu Alt Sınırı (referans): E ≈ {E_analytic_inf:.4f} eV")
print(f"(Sonlu kuyu için |E_sonlu| < V0={user_v0} ve E_sonlu > E_sonsuz_ref beklenir)\n")

# --- Görselleştirme ---
x_np   = x_plot.numpy().flatten()
# Potansiyel çizimi için smooth step
V_plot = -user_v0 * (1.0 / (1.0 + np.exp(-80.0 * (user_w / 2.0 - np.abs(x_np)))))

# Dalga fonksiyonunu potansiyel grafiğine sığdırmak için ölçekleme
max_psi = np.max(np.abs(psi_plot))
scale   = (user_v0 * 0.4) / max_psi if max_psi > 1e-8 else 1.0
psi_visual = psi_plot * scale + E_final

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Grafik: Potansiyel ve Dalga Fonksiyonu
ax1 = axes[0]
ax1.fill_between(x_np, V_plot, -user_v0 * 1.3, color="steelblue", alpha=0.15, label="Kuyu Bölgesi")
ax1.plot(x_np, V_plot, color="black", linewidth=2, label=f"V(x)  V0={user_v0}, W={user_w}")
ax1.axhline(E_final, color="red", linestyle="--", linewidth=2,
            label=f"E (PINN) = {E_final:.3f} eV")
ax1.axhline(E_analytic_inf, color="orange", linestyle=":", linewidth=1.5,
            label=f"E (∞ kuyu ref) = {E_analytic_inf:.3f} eV")
ax1.plot(x_np, psi_visual, color="royalblue", linewidth=2, label="ψ₀(x)  (ölçekli)")
ax1.set_title("Potansiyel ve Dalga Fonksiyonu", fontsize=13, fontweight="bold")
ax1.set_xlabel("Konum (x)"); ax1.set_ylabel("Enerji (eV)")
ax1.set_ylim(-user_v0 * 1.3, user_v0 * 0.3)
ax1.set_xlim(-L_domain, L_domain)
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# 2. Grafik: Olasılık Yoğunluğu
ax2 = axes[1]
prob  = psi_plot ** 2
dx_np = x_np[1] - x_np[0]
prob  = prob / (np.sum(prob) * dx_np)  # Olasılığı 1'e normalize et
ax2.plot(x_np, prob, color="darkorange", linewidth=2, label="|ψ₀(x)|²")
ax2.fill_between(x_np, prob, alpha=0.2, color="orange")
ax2.axvline(-user_w / 2, color="gray", linestyle=":", linewidth=1.5, label="Kuyu duvarları")
ax2.axvline( user_w / 2, color="gray", linestyle=":", linewidth=1.5)
ax2.set_title("Olasılık Yoğunluğu  |ψ₀(x)|²", fontsize=13, fontweight="bold")
ax2.set_xlabel("Konum (x)"); ax2.set_ylabel("|ψ|²")
ax2.set_xlim(-L_domain, L_domain)
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.suptitle(
    f"PINN Sonucu  |  V0={user_v0} eV,  W={user_w}  |  E = {E_final:.3f} eV",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("kuantum_sonuc.png", dpi=150, bbox_inches="tight")
plt.show()
print("Grafik 'kuantum_sonuc.png' olarak kaydedildi.")