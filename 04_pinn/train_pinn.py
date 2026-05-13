import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 0. CİHAZ (DEVICE) SEÇİMİ ---
# GPU yoksa CPU seçecektir. Gelecekte Colab gibi bir platformda çalıştırırsanız otomatik GPU kullanır.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

# --- 1. MİMARİ (SiLU Aktivasyonu ile Güncellendi) ---
class SurrogatePINN(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Tanh() yerine 2. türevleri çok daha iyi olan nn.SiLU() (Swish) kullanıyoruz
        self.net_psi = nn.Sequential(
            nn.Linear(3, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.net_E = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward_psi(self, x, V0, W):
        x_norm  = x  / 15.0
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0
        
        inp_pos = torch.cat([ x_norm, V0_norm, W_norm], dim=1)
        inp_neg = torch.cat([-x_norm, V0_norm, W_norm], dim=1)
        
        raw = (self.net_psi(inp_pos) + self.net_psi(inp_neg)) / 2.0
        envelope = 1.0 - (x_norm ** 2)
        return envelope * raw 

    def forward_E(self, V0, W):
        V0_norm = V0 / 50.0
        W_norm  = W  / 8.0
        raw_E = self.net_E(torch.cat([V0_norm, W_norm], dim=1))
        E_frac = -torch.sigmoid(raw_E) 
        return E_frac * V0


# W_t artık tensor kabul edecek şekilde güncellendi
def smooth_potential(x_col, V0_t, W_t, sharpness=80.0):
    half_w = W_t / 2.0
    inside = torch.sigmoid(sharpness * (half_w - torch.abs(x_col)))
    return -V0_t * inside


# --- 2. EĞİTİM HAZIRLIĞI ---
L_domain   = 15.0
N_far      = 100
N_near     = 200
BATCH_SIZE = 8   
EPOCHS     = 10000  # Vektörizasyon sayesinde çok daha hızlı bitecek

W_PDE  = 1.0
W_NORM = 100.0
W_E    = 1.0

model     = SurrogatePINN().to(device)
model     = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

print("=" * 60)
print(f"ADAM EĞİTİMİ BAŞLIYOR (Vektörize Edilmiş & Optimize Edilmiş)")
print("=" * 60)

start_time = time.time()

# --- 3. ADAM EĞİTİMİ (Döngüsüz, Vektörize) ---
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # Tüm batch için parametreleri tek seferde oluştur (FOR DÖNGÜSÜ KALDIRILDI)
    v0_vals = torch.empty(BATCH_SIZE, device=device).uniform_(2.0, 50.0)
    w_vals  = torch.empty(BATCH_SIZE, device=device).uniform_(1.0, 8.0)
    
    x_col_list = []
    v0_list = []
    w_list = []
    
    # Noktaları uzamsal olarak üretmek için hızlı liste
    for i in range(BATCH_SIZE):
        v0 = v0_vals[i]
        w  = w_vals[i]
        half_w = w / 2.0
        
        x_f = torch.empty(N_far, device=device).uniform_(-L_domain, L_domain)
        x_n = torch.empty(N_near, device=device).uniform_(-half_w.item() - 1.5, half_w.item() + 1.5)
        x_c = torch.cat([x_f, x_n])
        
        x_col_list.append(x_c)
        v0_list.append(torch.full_like(x_c, v0))
        w_list.append(torch.full_like(x_c, w))

    # Dev matrisler halinde birleştir
    x_col = torch.cat(x_col_list).unsqueeze(1).requires_grad_(True)
    V0_t  = torch.cat(v0_list).unsqueeze(1)
    W_t   = torch.cat(w_list).unsqueeze(1)

    # İleri Besleme (Tüm batch için TEK SEFERDE)
    psi    = model.forward_psi(x_col, V0_t, W_t)
    E_pred = model.forward_E(V0_t, W_t)

    # Türevler (Tüm batch için TEK SEFERDE)
    dpsi_dx = torch.autograd.grad(psi, x_col, torch.ones_like(psi), create_graph=True)[0]
    d2psi_dx2 = torch.autograd.grad(dpsi_dx, x_col, torch.ones_like(dpsi_dx), create_graph=True)[0]

    V_x = smooth_potential(x_col, V0_t, W_t)

    # Kayıplar
    residual = -0.5 * d2psi_dx2 + V_x * psi - E_pred * psi
    loss_pde = W_PDE * torch.mean(residual ** 2)
    loss_E   = W_E * torch.mean(E_pred)

    # Vektörize Normalizasyon
    x_grid_1d = torch.linspace(-L_domain, L_domain, 800, device=device).unsqueeze(1)
    x_grid    = x_grid_1d.repeat(BATCH_SIZE, 1)
    V0_grid   = v0_vals.repeat_interleave(800).unsqueeze(1)
    W_grid    = w_vals.repeat_interleave(800).unsqueeze(1)

    psi_g = model.forward_psi(x_grid, V0_grid, W_grid)
    psi_g_reshaped = psi_g.view(BATCH_SIZE, 800)
    dx_g = 2.0 * L_domain / 800
    norm_vals = torch.sum(psi_g_reshaped ** 2, dim=1) * dx_g
    loss_norm = W_NORM * torch.mean((norm_vals - 1.0) ** 2)

    # Toplam Loss ve Geri Yayılım
    total_loss = loss_pde + loss_norm + loss_E
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}/{EPOCHS} | Loss: {total_loss.item():.5f} | Ortalama E_pred: {E_pred.mean().item():.4f} eV")

print(f"Adam Eğitimi Tamamlandı. Geçen Süre: {(time.time() - start_time)/60:.1f} dakika.")

# --- 4. L-BFGS İLE İNCE AYAR (Fine-Tuning) ---
print("\n" + "=" * 60)
print("L-BFGS İNCE AYAR AŞAMASINA GEÇİLİYOR")
print("=" * 60)

# L-BFGS için sabit bir "zorlu" veri seti oluşturuyoruz (Dalgalanmayı önlemek için)
v0_fixed = torch.linspace(5.0, 45.0, BATCH_SIZE, device=device)
w_fixed  = torch.linspace(2.0, 7.0, BATCH_SIZE, device=device)

x_col_fix = []
v0_fix = []
w_fix = []
for i in range(BATCH_SIZE):
    x_f = torch.linspace(-L_domain, L_domain, N_far + N_near, device=device)
    x_col_fix.append(x_f)
    v0_fix.append(torch.full_like(x_f, v0_fixed[i]))
    w_fix.append(torch.full_like(x_f, w_fixed[i]))

x_col_lbfgs = torch.cat(x_col_fix).unsqueeze(1).requires_grad_(True)
V0_lbfgs    = torch.cat(v0_fix).unsqueeze(1)
W_lbfgs     = torch.cat(w_fix).unsqueeze(1)

# Normalizasyon gridi (L-BFGS için sabit)
x_grid_lbfgs  = x_grid_1d.repeat(BATCH_SIZE, 1)
V0_grid_lbfgs = v0_fixed.repeat_interleave(800).unsqueeze(1)
W_grid_lbfgs  = w_fixed.repeat_interleave(800).unsqueeze(1)

lbfgs_optimizer = torch.optim.LBFGS(
    model.parameters(), 
    lr=0.1, 
    max_iter=20, 
    tolerance_grad=1e-6, 
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe"
)
lbfgs_epochs = 100

def closure():
    lbfgs_optimizer.zero_grad()
    
    psi = model.forward_psi(x_col_lbfgs, V0_lbfgs, W_lbfgs)
    E_pred = model.forward_E(V0_lbfgs, W_lbfgs)
    
    dpsi_dx = torch.autograd.grad(psi, x_col_lbfgs, torch.ones_like(psi), create_graph=True)[0]
    d2psi_dx2 = torch.autograd.grad(dpsi_dx, x_col_lbfgs, torch.ones_like(dpsi_dx), create_graph=True)[0]
    V_x = smooth_potential(x_col_lbfgs, V0_lbfgs, W_lbfgs)
    
    loss_pde = W_PDE * torch.mean((-0.5 * d2psi_dx2 + V_x * psi - E_pred * psi) ** 2)
    loss_E   = W_E * torch.mean(E_pred)
    
    psi_g = model.forward_psi(x_grid_lbfgs, V0_grid_lbfgs, W_grid_lbfgs)
    psi_g_reshaped = psi_g.view(BATCH_SIZE, 800)
    norm_vals = torch.sum(psi_g_reshaped ** 2, dim=1) * dx_g
    loss_norm = W_NORM * torch.mean((norm_vals - 1.0) ** 2)
    
    loss = loss_pde + loss_norm + loss_E
    loss.backward()
    return loss

for epoch in range(lbfgs_epochs):
    loss_val = lbfgs_optimizer.step(closure)
    if epoch % 20 == 0:
        print(f"L-BFGS Adımı {epoch:3d} | Loss: {loss_val.item():.6f}")

# --- 5. KAYDET ---
# Kaydederken her zaman CPU tensorlarına çevirip kaydediyoruz ki app.py rahat okusun
torch.save(model.cpu().state_dict(), "kuantum_beyni.pth")
print("\nEğitim Başarıyla Bitti! Model 'kuantum_beyni.pth' dosyasına kaydedildi.")