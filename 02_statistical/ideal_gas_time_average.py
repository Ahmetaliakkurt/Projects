import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SİMÜLASYON PARAMETRELERİ
# ==========================================
N = 300           # Parçacık Sayısı
L = 10.0          # Kutu Boyutu (L x L) (2D'de Hacim = L^2'dir)
m = 1.0           # Parçacık Kütlesi
v_max = 5.0       # Başlangıç Maksimum Hızı
dt = 0.005         # Zaman Adımı
radius = 0.15     # Çarpışma Yarıçapı

# --- TERMODİNAMİK HEDEF (Sıcaklık Girdisi) ---
TARGET_T = 300.0             # Hedef Sıcaklık (Kelvin)
TARGET_KE = N * TARGET_T   # 2D için E_k = N * k_B * T (k_B = 1 alındı)

# Başlangıç Konumları ve Hızları
pos = np.random.uniform(radius, L - radius, size=(N, 2))
angles = np.random.uniform(0, 2 * np.pi, N)
speeds = np.random.uniform(1.0, v_max, N)
vel = np.zeros((N, 2))
vel[:, 0] = speeds * np.cos(angles)
vel[:, 1] = speeds * np.sin(angles)

# --- HIZ ÖLÇEKLENDİRME ---
current_random_ke = np.sum(0.5 * m * (vel[:, 0]**2 + vel[:, 1]**2))
scaling_factor = np.sqrt(TARGET_KE / current_random_ke)
vel[:, 0] *= scaling_factor
vel[:, 1] *= scaling_factor

initial_ke = TARGET_KE
P_theo = initial_ke / (L**2)

# İstatistik ve Çizim İçin Listeler
time_data = []
pressure_data = []          
instant_pressure_data = []  
ke_data = []
pe_data = []
te_data = []
error_data = []
accumulated_momentum = 0.0

x_max = 50.0 

# ==========================================
# 2. GRAFİK (FIGURE) KURULUMU
# ==========================================
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1.2, 0.6])

# SÜTUN 1: Gaz Kutusu ve Hata Grafiği
ax_box = fig.add_subplot(gs[0:2, 0])
ax_box.set_xlim(0, L)
ax_box.set_ylim(0, L)
ax_box.set_aspect('equal')
ax_box.set_xticks([])
ax_box.set_yticks([])
ax_box.set_title("İzole İdeal Gaz Sistemi", fontweight='bold')
for spine in ax_box.spines.values():
    spine.set_linewidth(2)

scatter = ax_box.scatter(pos[:, 0], pos[:, 1], s=20, c='blue', alpha=0.7, edgecolors='black')

ax_error = fig.add_subplot(gs[2, 0])
ax_error.set_title("Teorik Basınca Göre Bağıl Hata (%)", fontweight='bold')
ax_error.set_xlim(0, x_max)
ax_error.set_ylim(0, 100)
ax_error.set_xlabel("Zaman")
ax_error.set_ylabel("Hata (%)")
ax_error.grid(True, linestyle='--', alpha=0.6)
line_error, = ax_error.plot([], [], lw=2, color='magenta', label='Hata |Ö-T|/T')
ax_error.legend(loc="upper right")
error_text = ax_error.text(0.05, 0.15, "", transform=ax_error.transAxes, fontsize=14, fontweight='bold', color='darkred')

# HUD (Artık T değerini baz alıyor)
info_text = (f"PARAMETRELER\n"
             f"------------------\n"
             f"N  = {N}\n"
             f"L  = {L} m\n"
             f"m  = {m} kg\n\n"
             f"SABİT SICAKLIK\n"
             f"T = {TARGET_T:.2f} K\n\n"
             f"TEORİK BASINÇ\n"
             f"P_theo = {P_theo:.3f}")

ax_box.text(0.03, 0.96, info_text, transform=ax_box.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# SÜTUN 2: Termodinamik Grafikler
ax_press = fig.add_subplot(gs[0, 1])
ax_press.set_title(f"Zaman Ortalamalı Basınç (Hedef: {P_theo:.3f})", fontweight='bold')
ax_press.set_xlim(0, x_max)
ax_press.set_ylim(0, P_theo * 2) 
ax_press.set_ylabel("Basınç (P)")
ax_press.grid(True, linestyle='--', alpha=0.6)
line_press, = ax_press.plot([], [], lw=2, color='darkred', label='Ölçülen P')
ax_press.axhline(P_theo, color='blue', linestyle='--', alpha=0.7, label='Teorik P')
ax_press.legend(loc="upper right")

ax_hist = fig.add_subplot(gs[1, 1])
ax_hist.set_title("Basınç Olasılık Dağılımı", fontweight='bold')
ax_hist.set_xlabel("Anlık Basınç (P)")
ax_hist.set_ylabel("Olasılık")
ax_hist.grid(True, linestyle='--', alpha=0.6)

ax_energy = fig.add_subplot(gs[2, 1])
ax_energy.set_title("Sistem Enerjisi", fontweight='bold')
ax_energy.set_xlim(0, x_max)
ax_energy.set_ylim(0, initial_ke + 100) 
ax_energy.set_xlabel("Zaman")
ax_energy.set_ylabel("Enerji (J)")
ax_energy.grid(True, linestyle='--', alpha=0.6)
line_ke, = ax_energy.plot([], [], lw=2, color='green', label='Kinetik (KE)')
line_pe, = ax_energy.plot([], [], lw=2, color='orange', label='Potansiyel (PE)')
line_te, = ax_energy.plot([], [], lw=2, linestyle=':', color='black', label='Toplam (TE)')
ax_energy.legend(loc="right")

# SÜTUN 3: SADECE V, N, T TABLOSU
ax_dash = fig.add_subplot(gs[:, 2])
ax_dash.axis('off') 
dash_text = ax_dash.text(0.1, 0.90, "", transform=ax_dash.transAxes, 
                         fontsize=14, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round,pad=1', facecolor='#1e1e1e', edgecolor='#444444', alpha=0.9),
                         color='#00ff00')

fig.tight_layout()

# ==========================================
# 3. SİMÜLASYON MOTORU
# ==========================================
def animate(frame):
    global pos, vel, accumulated_momentum, x_max
    
    current_time = (frame + 1) * dt
    time_data.append(current_time)
    
    pos += vel * dt
    
    hit_left = pos[:, 0] <= radius
    hit_right = pos[:, 0] >= L - radius
    hit_bottom = pos[:, 1] <= radius
    hit_top = pos[:, 1] >= L - radius
    
    v_left = np.abs(vel[hit_left, 0])
    v_right = np.abs(vel[hit_right, 0])
    v_bottom = np.abs(vel[hit_bottom, 1])
    v_top = np.abs(vel[hit_top, 1])
    
    dp_instant = np.sum(2 * m * v_left) + np.sum(2 * m * v_right) + \
                 np.sum(2 * m * v_bottom) + np.sum(2 * m * v_top)
               
    accumulated_momentum += dp_instant
    
    vel[hit_left, 0] = np.abs(vel[hit_left, 0])
    vel[hit_right, 0] = -np.abs(vel[hit_right, 0])
    vel[hit_bottom, 1] = np.abs(vel[hit_bottom, 1])
    vel[hit_top, 1] = -np.abs(vel[hit_top, 1])
    
    scatter.set_offsets(pos)
    
    current_pressure = accumulated_momentum / (current_time * (4 * L))
    pressure_data.append(current_pressure)
    
    current_error = (np.abs(current_pressure - P_theo) / P_theo) * 100.0
    error_data.append(current_error)
    
    instant_pressure = dp_instant / (dt * 4 * L)
    if instant_pressure > 0:
        instant_pressure_data.append(instant_pressure)
    
    current_ke = np.sum(0.5 * m * (vel[:, 0]**2 + vel[:, 1]**2))
    current_pe = 0.0 
    
    ke_data.append(current_ke)
    pe_data.append(current_pe)
    te_data.append(current_ke + current_pe)
    
    # --- MAKROSKOBİK PARAMETRELER (V, N, T) ---
    current_temp = current_ke / N  # T = E_k / N
    current_vol = L**2             # 2D Hacim (Alan)
    
    if current_time >= x_max:
        x_max += 50.0
        ax_press.set_xlim(0, x_max)
        ax_energy.set_xlim(0, x_max)
        ax_error.set_xlim(0, x_max)
        
    if frame > 10 and max(pressure_data[-50:]) > ax_press.get_ylim()[1] * 0.9:
        ax_press.set_ylim(0, max(pressure_data[-50:]) * 1.2)
        
    if frame > 20:
        current_max_error = np.max(error_data[-50:])
        ax_error.set_ylim(0, max(5.0, current_max_error * 1.2))
        
    line_press.set_data(time_data, pressure_data)
    line_ke.set_data(time_data, ke_data)
    line_pe.set_data(time_data, pe_data)
    line_te.set_data(time_data, te_data)
    line_error.set_data(time_data, error_data)
    
    error_text.set_text(f"Anlık Hata: % {current_error:.2f}")
    
    # SADECE V, N, T İÇEREN TABLO
    dashboard_str = (
        f"MAKROSKOBİK TABLO\n"
        f"+-------+----------+\n"
        f"| Param | Değer    |\n"
        f"+-------+----------+\n"
        f"|   N   | {N:<8} |\n"
        f"|   V   | {current_vol:<4.1f} m²  |\n"
        f"|   T   | {current_temp:<5.2f} K  |\n"
        f"+-------+----------+\n"
    )
    dash_text.set_text(dashboard_str)
    
    if frame % 10 == 0 and len(instant_pressure_data) > 5:
        ax_hist.clear()
        ax_hist.hist(instant_pressure_data, bins=25, density=True, color='purple', alpha=0.7, edgecolor='black')
        ax_hist.set_title("Basınç Olasılık Dağılımı", fontweight='bold')
        ax_hist.set_xlabel("Anlık Basınç (P)")
        ax_hist.set_ylabel("Olasılık Yoğunluğu")
        ax_hist.grid(True, linestyle='--', alpha=0.6)
        
        mean_p = np.mean(instant_pressure_data)
        ax_hist.axvline(mean_p, color='red', linestyle='dashed', linewidth=2, label=f'Ortalama: {mean_p:.1f}')
        ax_hist.legend(loc="upper right")
    
    return scatter, line_press, line_ke, line_pe, line_te, line_error, error_text, dash_text

ani = animation.FuncAnimation(fig, animate, frames=itertools.count(), interval=30, blit=False)
plt.show()