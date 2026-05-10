import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. KULLANICIDAN SEVİYE SEÇİMİ ALMA ---
print("-" * 50)
print("PINN Kuantum Kuyusu Çözücüsüne Hoş Geldiniz!")
print("Bu kuyu (V0=10, Width=2) sadece 3 adet bağlı duruma sahiptir: 0, 1 ve 2.")
print("-" * 50)

while True:
    try:
        state_choice = int(input("Hangi enerji seviyesini çözmek istersiniz? (0, 1 veya 2): "))
        if state_choice in [0, 1, 2]:
            break
        else:
            print("Hata: Lütfen sadece 0, 1 veya 2 değerlerinden birini girin.")
    except ValueError:
        print("Hata: Geçerli bir tam sayı giriniz.")

# --- 2. FİZİKSEL PARAMETRELER ---
V0 = 10.0
WIDTH = 2.0
w_param = WIDTH / 2.0
L_domain = 4.0

# Seçilen seviyeye göre Yapay Zekaya "Fiziksel İpuçları" veriyoruz
if state_choice == 0:
    e_init_val = -7.0      # State 0 tahmini enerjisi (Kuyu tabanına yakın)
    is_even_parity = True  # State 0 simetriktir (Çift Fonksiyon)
elif state_choice == 1:
    e_init_val = -4.0      # State 1 tahmini enerjisi
    is_even_parity = False # State 1 asimetriktir (Tek Fonksiyon)
elif state_choice == 2:
    e_init_val = -2.0      # State 2 tahmini enerjisi (Kuyu ağzına yakın)
    is_even_parity = True  # State 2 yine simetriktir (Çift Fonksiyon)

# --- 3. PINN MİMARİSİ ---
class QuantumPINN(nn.Module):
    def __init__(self, e_init):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Ağırlıkların düzgün (dik) başlatılması
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                # Simetrinin kırılması için küçük rastgele bias değerleri veriyoruz
                nn.init.normal_(layer.bias, mean=0.0, std=0.1)

        # Ağın enerji tahmini buradan başlar
        self.E = nn.Parameter(torch.tensor([e_init])) 

    def forward(self, x):
        # Ağa seviyenin simetrisini (Parite) zorunlu kılıyoruz
        if is_even_parity:
            # Çift Fonksiyon: Temel Durum (0) ve State 2 için
            return (self.net(x) + self.net(-x)) / 2.0
        else:
            # Tek Fonksiyon: State 1 için (Orijinde tam sıfır olmasını garanti eder)
            return (self.net(x) - self.net(-x)) / 2.0

def V_potential(x):
    # Kare kuyu potansiyeli
    return torch.where(torch.abs(x) < w_param, torch.tensor(-V0), torch.tensor(0.0))

# --- 4. EĞİTİM HAZIRLIĞI ---
model = QuantumPINN(e_init_val)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
x_collocation = torch.linspace(-L_domain, L_domain, 1000).view(-1, 1)
x_collocation.requires_grad = True 
dx = (2 * L_domain) / 1000.0
epochs = 7000 

print(f"\n[State {state_choice}] için PINN Eğitimi Başlıyor (Kaçış yok!)...")

# --- 5. EĞİTİM DÖNGÜSÜ ---
for epoch in range(epochs):
    optimizer.zero_grad()
    psi = model(x_collocation)
    E_pred = model.E

    # Türevler ve Schrödinger Denklemi (PDE Loss)
    dpsi_dx = torch.autograd.grad(psi, x_collocation, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    d2psi_dx2 = torch.autograd.grad(dpsi_dx, x_collocation, grad_outputs=torch.ones_like(dpsi_dx), create_graph=True)[0]
    
    V_x = V_potential(x_collocation)
    physics_residual = -0.5 * d2psi_dx2 + V_x * psi - E_pred * psi
    loss_pde = torch.mean(physics_residual**2)

    # Sınır Koşulları (Boundary Loss) - 100 ile çarpılarak vurgulandı
    x_left = torch.tensor([[-L_domain]])
    x_right = torch.tensor([[L_domain]])
    loss_bc = (model(x_left)**2 + model(x_right)**2) * 100.0

    # Normalizasyon (Olasılık Loss) - Ağın tembellik yapmasını engellemek için 1000 ile çarpıldı
    integral = torch.sum(psi**2) * dx
    loss_norm = ((integral - 1.0)**2) * 1000.0

    # Toplam Ceza
    total_loss = loss_pde + loss_bc[0][0] + loss_norm

    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss = {total_loss.item():.5f} | Enerji (E) = {E_pred.item():.4f}")

print(f"\nEğitim Tamamlandı! Bulunan State {state_choice} Enerjisi: {model.E.item():.4f}")

# --- 6. GÖRSELLEŞTİRME ---
x_plot = x_collocation.detach().numpy()
psi_plot = model(x_collocation).detach().numpy()
V_plot = V_potential(x_collocation).detach().numpy()
E_final = model.E.item()

# ---------------------------------------------------------
# SİHİRLİ DOKUNUŞ: Kuantum Fazı (İşaret) Düzeltmesi
# Eğer ağ rastgele olarak negatif (-psi) çözüme yakınsadıysa, 
# görsel standartlara uyması için dalgayı ters çeviriyoruz.
if state_choice == 0 or state_choice == 2:
    # Çift fonksiyonlar: Tam ortadaki (x=0) tepe negatifse, komple -1 ile çarp
    center_idx = len(psi_plot) // 2
    if psi_plot[center_idx] < 0:
        psi_plot = -psi_plot
elif state_choice == 1:
    # Tek fonksiyon: Sağ taraftaki (x>0) tepe negatifse, komple -1 ile çarp
    right_peak_idx = int(len(psi_plot) * 0.65) # x'in pozitif olduğu bölge
    if psi_plot[right_peak_idx] < 0:
        psi_plot = -psi_plot
# ---------------------------------------------------------

# Dalga Fonksiyonunu Görünür Kılmak İçin Büyütme (Scaling) İşlemi
max_amp = np.max(np.abs(psi_plot))
if max_amp > 0:
    visual_scale = 1.5 / max_amp  # Dalganın tepe noktasını 1.5 eV yüksekliğine sabitler
else:
    visual_scale = 1.0

psi_visual = (psi_plot * visual_scale) + E_final

plt.figure(figsize=(10, 6))
plt.plot(x_plot, V_plot, color='black', linewidth=2, alpha=0.5, label='Potansiyel V(x)')
plt.fill_between(x_plot.flatten(), V_plot.flatten(), -V0*1.2, color='gray', alpha=0.2)

# Enerji Seviyesi Çizgisi
plt.hlines(E_final, -L_domain, L_domain, color='red', linestyle='--', linewidth=1.5, label=f'State {state_choice} Enerjisi (E={E_final:.2f} eV)')

# Dalga Fonksiyonu Çizgisi
plt.plot(x_plot, psi_visual, color='blue', linewidth=2, label=f'State {state_choice} Dalga Fonksiyonu ($\psi$)')

# Olasılık Gölgelendirmesi
plt.fill_between(x_plot.flatten(), E_final, psi_visual.flatten(), where=(psi_visual.flatten() > E_final), color='blue', alpha=0.2)
plt.fill_between(x_plot.flatten(), E_final, psi_visual.flatten(), where=(psi_visual.flatten() <= E_final), color='blue', alpha=0.2)

plt.title(f"PINN ile Çözülmüş Kuantum Kuyusu (State {state_choice})", fontsize=14, fontweight='bold')
plt.xlabel("Konum (x)", fontsize=12)
plt.ylabel("Enerji (E)", fontsize=12)
plt.ylim(-V0 * 1.2, max(E_final + 2.5, 0))
plt.xlim(-L_domain, L_domain)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()