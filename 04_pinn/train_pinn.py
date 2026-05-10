import torch
import torch.nn as nn
import numpy as np

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
                nn.init.normal_(layer.bias, mean=0.0, std=0.1)
                
        for layer in self.net_E:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))

    def forward_psi(self, x, V0, W):
        # MUHTEŞEM ÇÖZÜM: NORMALİZASYON!
        # Ağı felçten kurtarmak için tüm girdileri -1 ile 1 arasına sıkıştırıyoruz.
        x_norm = x / 10.0
        V0_norm = V0 / 50.0
        W_norm = W / 8.0

        inputs_pos = torch.cat([x_norm, V0_norm, W_norm], dim=1)
        inputs_neg = torch.cat([-x_norm, V0_norm, W_norm], dim=1)
        
        psi_pos = self.net_psi(inputs_pos)
        psi_neg = self.net_psi(inputs_neg)
        return torch.nn.functional.softplus((psi_pos + psi_neg) / 2.0)

    def forward_E(self, V0, W):
        # Enerji ağı için de girdileri normalize ediyoruz!
        V0_norm = V0 / 50.0
        W_norm = W / 8.0
        raw_E = self.net_E(torch.cat([V0_norm, W_norm], dim=1))
        # Çıktıyı fiziksel sınırlara (-V0 ile 0 arası) yayıyoruz
        return -V0 * torch.sigmoid(raw_E)

# --- 2. EĞİTİM HAZIRLIĞI ---
L_domain = 10.0
model = SurrogatePINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x_collocation = torch.linspace(-L_domain, L_domain, 800).view(-1, 1)
x_collocation.requires_grad = True 
dx = (2 * L_domain) / 800.0

epochs = 15000 
print("Surrogate PINN Eğitimi Başlıyor (Kaybolan Gradyanlar düzeltildi!)...")

# --- 3. EĞİTİM DÖNGÜSÜ (Mini-Batch Eklendi) ---
for epoch in range(epochs):
    optimizer.zero_grad()
    total_batch_loss = 0
    
    # Ağın kafası karışmasın diye her adımda 4 farklı kuyu gösteriyoruz
    for _ in range(4):
        v0_val = np.random.uniform(1.0, 50.0)
        w_val = np.random.uniform(1.0, 8.0)
        
        V0_t = torch.full_like(x_collocation, v0_val)
        W_t = torch.full_like(x_collocation, w_val)

        psi = model.forward_psi(x_collocation, V0_t, W_t)
        E_pred = model.forward_E(torch.tensor([[v0_val]]), torch.tensor([[w_val]]))

        dpsi_dx = torch.autograd.grad(psi, x_collocation, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        d2psi_dx2 = torch.autograd.grad(dpsi_dx, x_collocation, grad_outputs=torch.ones_like(dpsi_dx), create_graph=True)[0]
        
        V_x = torch.where(torch.abs(x_collocation) < w_val/2.0, -V0_t, torch.zeros_like(x_collocation))
        
        physics_residual = -0.5 * d2psi_dx2 + V_x * psi - E_pred * psi
        loss_pde = torch.mean(physics_residual**2)

        x_left, x_right = torch.tensor([[-L_domain]]), torch.tensor([[L_domain]])
        v0_bc, w_bc = torch.tensor([[v0_val]]), torch.tensor([[w_val]])
        loss_bc = (model.forward_psi(x_left, v0_bc, w_bc)**2 + model.forward_psi(x_right, v0_bc, w_bc)**2) * 100.0
        
        loss_norm = (((torch.sum(psi**2) * dx) - 1.0)**2) * 1000.0

        total_batch_loss += (loss_pde + loss_bc[0][0] + loss_norm)

    # 4 kuyunun ortalama hatasına göre ağırlıkları güncelle
    total_loss = total_batch_loss / 4.0
    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Örnek Kuyu: V0={v0_val:.1f}, W={w_val:.1f} | Loss = {total_loss.item():.4f} | E_Tahmin = {E_pred.item():.2f}")

# --- 4. MODELİ KAYDETME ---
torch.save(model.state_dict(), "kuantum_beyni.pth")
print("\nEğitim Bitti! Yapay zekanın beyni 'kuantum_beyni.pth' dosyasına kaydedildi.")