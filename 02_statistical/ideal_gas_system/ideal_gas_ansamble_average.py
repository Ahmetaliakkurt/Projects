import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. PARAMETRELER
# ==========================================
G = 9.81    
L1, L2 = 1.0, 1.0  
M1, M2 = 1.0, 1.0  

N_pendulums = 200           # Entropiyi daha hassas ölçmek için sayıyı 200 yaptık
start_angle_deg = 179.0     
noise_std_deg = 1e-6        

T_sim = 20.0                
fps = 30
frames = int(T_sim * fps)
t_eval = np.linspace(0, T_sim, frames)

print("Kaotik Diferansiyel Denklemler Çözülüyor...")

def double_pendulum_derivs(t, state):
    state = state.reshape(-1, 4)
    t1, w1, t2, w2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    delta = t1 - t2
    den = (2*M1 + M2 - M2 * np.cos(2*t1 - 2*t2))
    
    dw1 = (-G * (2*M1 + M2) * np.sin(t1) - M2 * G * np.sin(t1 - 2*t2) 
           - 2 * np.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * np.cos(delta))) / (L1 * den)
    dw2 = (2 * np.sin(delta) * (w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(t1) 
           + w2**2 * L2 * M2 * np.cos(delta))) / (L2 * den)
    return np.stack([w1, dw1, w2, dw2], axis=1).flatten()

np.random.seed(42)
y0 = []
theta1_initials = np.random.normal(loc=np.radians(start_angle_deg), scale=np.radians(noise_std_deg), size=N_pendulums)
for th1 in theta1_initials: y0.extend([th1, 0.0, th1, 0.0])

sol = solve_ivp(double_pendulum_derivs, (0, T_sim), y0, t_eval=t_eval, method='RK45')
sol_y = sol.y

print("Veriler İşleniyor (Shannon Entropisi Hesaplanıyor)...")

# ==========================================
# 2. İSTATİSTİKSEL HESAPLAMALAR
# ==========================================
theta2_all = sol_y[2::4, :] 
theta2_wrapped = np.arctan2(np.sin(theta2_all), np.cos(theta2_all)) # [-pi, pi] arası

# 2.A: Standart Sapma
std_dev_deg = np.degrees(np.std(theta2_wrapped, axis=0))

# 2.B: Shannon Entropisi Hesabı (Histogram Yöntemi)
# Çemberi 36 dilime (10'ar derecelik bins) bölerek "olasılık p(x)" buluyoruz
bins = 36 
entropy_bits = np.zeros(frames)
for i in range(frames):
    counts, _ = np.histogram(theta2_wrapped[:, i], bins=bins, range=(-np.pi, np.pi))
    probabilities = counts / N_pendulums
    # Logaritma 0 hatasını önlemek için > 0 olanları al
    p_nonzero = probabilities[probabilities > 0]
    entropy_bits[i] = -np.sum(p_nonzero * np.log2(p_nonzero))

# Maksimum teorik Shannon Entropisi: Düzgün dağılım (log2(N_bins))
max_entropy = np.log2(bins) 

# ==========================================
# 3. ÇİFT EKSENLİ GÖRSELLEŞTİRME
# ==========================================
fig, (ax_pend, ax_stat) = plt.subplots(1, 2, figsize=(16, 7))
fig.canvas.manager.set_window_title('Kaos: Standart Sapma vs Shannon Entropisi')
fig.patch.set_facecolor('#121212')

# --- Sol Panel (Sarkaçlar) ---
ax_pend.set_facecolor('black')
ax_pend.set_xlim(-2.5, 2.5); ax_pend.set_ylim(-2.5, 2.5); ax_pend.set_aspect('equal')
ax_pend.set_title("Faz Uzayı: 200 Sarkaç", color='white')
lines = [ax_pend.plot([], [], 'o-', lw=1, color=plt.cm.plasma(i/N_pendulums), alpha=0.3, markersize=1)[0] for i in range(N_pendulums)]

# --- Sağ Panel (Çift Eksenli İstatistik) ---
ax_stat.set_facecolor('black')
ax_stat.set_xlim(0, T_sim)
ax_stat.set_xlabel("Zaman (s)", color='white')

# 1. EKSEN: Standart Sapma (Fiziksel Dağılım)
ax_stat.set_ylabel("Standart Sapma (Derece)", color='magenta')
ax_stat.tick_params(axis='y', colors='magenta', labelcolor='magenta')
ax_stat.set_ylim(-5, 120)
line_std, = ax_stat.plot([], [], color='magenta', lw=2, label='Standart Sapma (σ)')

# 2. EKSEN: Shannon Entropisi (Bilgi Kaybı)
ax_ent = ax_stat.twinx()  # Aynı X eksenini paylaşan ikinci Y ekseni
ax_ent.set_ylabel("Shannon Entropisi (Bit)", color='cyan')
ax_ent.tick_params(axis='y', colors='cyan', labelcolor='cyan')
ax_ent.set_ylim(-0.2, max_entropy * 1.1)
line_ent, = ax_ent.plot([], [], color='cyan', lw=3, label='Entropi (S)')
ax_ent.axhline(max_entropy, color='cyan', linestyle=':', lw=2, alpha=0.6, label=f'Max Entropi ({max_entropy:.2f} Bit)')

time_vline = ax_stat.axvline(0, color='white', linestyle='--', lw=1)

# Efsaneleri birleştir
lines_leg, labels_leg = ax_stat.get_legend_handles_labels()
lines_leg2, labels_leg2 = ax_ent.get_legend_handles_labels()
ax_stat.legend(lines_leg + lines_leg2, labels_leg + labels_leg2, loc='lower right', facecolor='black', labelcolor='white')

time_text = ax_pend.text(0.05, 0.95, '', transform=ax_pend.transAxes, color='white', fontsize=12)

def update(frame):
    for i in range(N_pendulums):
        th1, th2 = sol_y[i*4][frame], sol_y[i*4 + 2][frame]
        x1, y1 = L1 * np.sin(th1), -L1 * np.cos(th1)
        x2, y2 = x1 + L2 * np.sin(th2), y1 - L2 * np.cos(th2)
        lines[i].set_data([0, x1, x2], [0, y1, y2])
    
    line_std.set_data(t_eval[:frame], std_dev_deg[:frame])
    line_ent.set_data(t_eval[:frame], entropy_bits[:frame])
    time_vline.set_xdata([t_eval[frame], t_eval[frame]])
    
    time_text.set_text(f'Zaman: {t_eval[frame]:.2f} s')
    return lines + [line_std, line_ent, time_vline, time_text]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)
plt.tight_layout()
plt.show()