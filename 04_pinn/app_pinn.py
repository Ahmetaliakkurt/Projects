import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys

# Eğitilen modelin mimarisini tanıtıyoruz (Sadece iskelet)
class SurrogatePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_psi = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.net_E = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )
    def forward_psi(self, x, V0, W):
        inputs_pos = torch.cat([x, V0, W], dim=1)
        inputs_neg = torch.cat([-x, V0, W], dim=1)
        return torch.nn.functional.softplus((self.net_psi(inputs_pos) + self.net_psi(inputs_neg)) / 2.0)
    def forward_E(self, V0, W):
        return -V0 * torch.sigmoid(self.net_E(torch.cat([V0, W], dim=1)))

print("=" * 60)
print("HIZLI KUANTUM ÇÖZÜCÜYE HOŞ GELDİNİZ (Yapay Zeka Destekli)")
print("=" * 60)

# Modeli Yükleme
model = SurrogatePINN()
try:
    model.load_state_dict(torch.load("kuantum_beyni.pth", weights_only=True))
    model.eval() # Eğitimi kapat, sadece tahmin yap
except FileNotFoundError:
    print("HATA: 'kuantum_beyni.pth' bulunamadı. Önce eğitim kodunu çalıştırın!")
    sys.exit()

# Kullanıcıdan Değer Alma
while True:
    try:
        user_v0 = float(input("Kuyu Derinliğini girin (0 ile 50 arası): "))
        user_w = float(input("Kuyu Genişliğini girin (1 ile 8 arası): "))
        if 0 < user_v0 <= 50 and 1 <= user_w <= 8:
            break
        print("Lütfen belirtilen aralıklarda değer girin.")
    except ValueError:
        print("Lütfen sayısal bir değer girin.")

# SIFIR DENKLEM ÇÖZÜMÜ - Sadece Yapay Zekaya Soruyoruz!
L_domain = 10.0
x_plot = torch.linspace(-L_domain, L_domain, 1000).view(-1, 1)
V0_t = torch.full_like(x_plot, user_v0)
W_t = torch.full_like(x_plot, user_w)

# Saniyesinde Cevabı Al
with torch.no_grad():
    psi_plot = model.forward_psi(x_plot, V0_t, W_t).numpy()
    E_final = model.forward_E(torch.tensor([[user_v0]]), torch.tensor([[user_w]])).item()

# Görselleştirme
x_np = x_plot.numpy()
V_plot = np.where(np.abs(x_np) < user_w / 2.0, -user_v0, 0.0)

visual_scale = 1.5 / np.max(np.abs(psi_plot)) if np.max(np.abs(psi_plot)) > 0 else 1.0
psi_visual = (psi_plot * visual_scale) + E_final

plt.figure(figsize=(10, 6))
plt.plot(x_np, V_plot, color='black', linewidth=2, alpha=0.5, label=f'Potansiyel (V0={user_v0}, W={user_w})')
plt.fill_between(x_np.flatten(), V_plot.flatten(), -user_v0*1.2, color='gray', alpha=0.2)
plt.hlines(E_final, -L_domain, L_domain, color='red', linestyle='--', linewidth=2, label=f'Tahmin Edilen Enerji: {E_final:.3f} eV')
plt.plot(x_np, psi_visual, color='blue', linewidth=2, label='Dalga Fonksiyonu ($\psi_0$)')

plt.title(f"Yapay Zeka Anlık Çözümü (E = {E_final:.3f})", fontsize=14, fontweight='bold')
plt.xlabel("Konum (x)")
plt.ylabel("Enerji (E)")
plt.ylim(-user_v0 * 1.2, max(E_final + user_v0*0.2, 0))
plt.xlim(-L_domain, L_domain)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
