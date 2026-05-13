import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. PARAMETRELER (Büyük Kutu, Küçük Paket)
# ==========================================
L = 10.0           # Kutu boyutu (Genişletildi)
N = 500            # Grid çözünürlüğü (Pürüzsüzlük için artırıldı)
n_max = 120        # Baz fonksiyonu sayısı (Küçük paket için yüksek tutulmalı)
hbar = 1.0
mass = 1.0

# Küçük Gaussian Paket Parametreleri
x0, y0 = 5.0, 5.0    # Başlangıç konumu
sigma = 0.35         # Paket genişliği (Daha küçük/keskin)
kx0, ky0 = 15.0, 10.0 # Başlangıç momentumu

x_vec = np.linspace(0, L, N)
y_vec = np.linspace(0, L, N)
X, Y = np.meshgrid(x_vec, y_vec)

# ==========================================
# 2. SPEKTRAL ANALİZ (Analitik Çözüm)
# ==========================================
# İlk Durum: Gaussian
psi_0 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * \
        np.exp(1j * (kx0 * X + ky0 * Y))

# Normalizasyon
psi_0 /= np.sqrt(np.sum(np.abs(psi_0)**2) * (L/N)**2)

print(f"Spektral katsayılar hesaplanıyor (n_max={n_max})...")

# Baz fonksiyonları (Sinüsler) ve Katsayılar (DST-II benzeri projeksiyon)
n_vals = np.arange(1, n_max + 1)
# Matris çarpımı ile hızlı projeksiyon: C = <phi_nm | psi_0>
sin_X = np.sin(np.outer(x_vec, n_vals) * np.pi / L)
sin_Y = np.sin(np.outer(y_vec, n_vals) * np.pi / L)

# Spektral katsayı matrisi (n_max x n_max)
# Katsayılar = (Integral psi_0 * sin_n * sin_m)
C = (sin_Y.T @ psi_0 @ sin_X) * (4.0 / N**2) 

# Enerji Seviyeleri: E_nm = (hbar^2 * pi^2 / 2mL^2) * (n^2 + m^2)
E_scale = (np.pi**2 * hbar**2) / (2 * mass * L**2)
E_n = n_vals**2
E_nm = np.add.outer(E_n, E_n) * E_scale

def get_psi_at_t(t):
    """Zaman evrimi spektral uzayda faz kaydırmadır."""
    # psi(t) = sum C_nm * exp(-i E_nm t / hbar) * phi_nm
    C_t = C * np.exp(-1j * (E_nm / hbar) * t)
    return sin_Y @ C_t @ sin_X.T

# ==========================================
# 3. GÖRSELLEŞTİRME
# ==========================================
fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor('#050505')
ax.set_facecolor('#050505')

psi_init = get_psi_at_t(0)
im = ax.imshow(np.abs(psi_init)**2, extent=[0, L, 0, L], origin='lower',
               cmap='magma', interpolation='spline36') # bicubic ile ekstra pürüzsüzlük

ax.set_title("2D Spektral Kuantum Bilardo", color='white', fontsize=12)
ax.axis('off')

# Kenarlar (Sonsuz Potansiyel Duvarları)
rect = plt.Rectangle((0, 0), L, L, linewidth=2, edgecolor='#00ffcc', facecolor='none')
ax.add_patch(rect)

dt = 0.015

def update(frame):
    t = frame * dt
    psi_t = get_psi_at_t(t)
    prob = np.abs(psi_t)**2
    
    im.set_array(prob)
    # Dinamik kontrast ayarı: Paket yayıldıkça detayları görmek için
    im.set_clim(0, np.max(prob) * 0.7)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=600, interval=25, blit=True)
plt.show()