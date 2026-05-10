import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

# --- Fiziksel Parametreler ---
G = 9.81    
L1, L2 = 1.0, 1.0  
M1, M2 = 1.0, 1.0  

def double_pendulum_derivs(t, state):
    # State: [t1_n, w1_n, t2_n, w2_n, ...] şeklinde 4*N boyutlu vektör
    state = state.reshape(-1, 4)
    t1, w1, t2, w2 = state[:,0], state[:,1], state[:,2], state[:,3]
    
    delta = t1 - t2
    den = (2*M1 + M2 - M2 * np.cos(2*t1 - 2*t2))
    
    dw1 = (-G * (2*M1 + M2) * np.sin(t1) 
           - M2 * G * np.sin(t1 - 2*t2) 
           - 2 * np.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * np.cos(delta))) / (L1 * den)
    
    dw2 = (2 * np.sin(delta) * (w1**2 * L1 * (M1 + M2) 
           + G * (M1 + M2) * np.cos(t1) 
           + w2**2 * L2 * M2 * np.cos(delta))) / (L2 * den)
    
    return np.stack([w1, dw1, w2, dw2], axis=1).flatten()

# --- Görselleştirme Hazırlığı ---
fig, ax_sim = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(top=0.85, bottom=0.2)

ax_sim.set_aspect('equal')
ax_sim.set_xlim(-2.2, 2.2)
ax_sim.set_ylim(-2.2, 2.2)
ax_sim.grid(True, alpha=0.2, linestyle='--')

# Kontrol değişkenleri
is_running = False
ani = None
lines = []

def run_simulation(num_n):
    global ani, lines, is_running
    is_running = False # Yeni simülasyon kurulurken durdur
    ax_sim.clear()
    ax_sim.set_xlim(-2.2, 2.2); ax_sim.set_ylim(-2.2, 2.2); ax_sim.grid(True, alpha=0.2)
    
    # Başlangıç Koşulları
    t1_base = np.pi/2 # 90 derece
    t2_base = np.pi   # 180 derece
    y0 = []
    
    for i in range(num_n):
        # Her sarkaç bir öncekinden 1 derece daha saat yönünde (negatif yönde) başlar
        offset = np.radians(i)
        y0.extend([t1_base - offset, 0, t2_base, 0])
    
    t_eval = np.linspace(0, 30, 1500)
    sol = solve_ivp(double_pendulum_derivs, (0, 30), y0, t_eval=t_eval, method='RK45')
    
    colors = plt.cm.plasma(np.linspace(0, 1, num_n))
    lines = [ax_sim.plot([], [], 'o-', lw=2, color=colors[i], alpha=0.6)[0] for i in range(num_n)]
    
    # Zaman göstergesi
    time_text = ax_sim.text(0.05, 0.95, '', transform=ax_sim.transAxes, fontweight='bold')

    def update(frame):
        if not is_running: 
            return lines + [time_text]
        
        for i in range(num_n):
            idx = i * 4
            theta1, theta2 = sol.y[idx][frame], sol.y[idx+2][frame]
            
            x1 = L1 * np.sin(theta1)
            y1 = -L1 * np.cos(theta1)
            x2 = x1 + L2 * np.sin(theta2)
            y2 = y1 - L2 * np.cos(theta2)
            
            lines[i].set_data([0, x1, x2], [0, y1, y2])
        
        time_text.set_text(f'Zaman: {t_eval[frame]:.1f}s')
        return lines + [time_text]

    # Başlangıç pozisyonunu çiz
    for i in range(num_n):
        idx = i * 4
        theta1, theta2 = sol.y[idx][0], sol.y[idx+2][0]
        x1, y1 = L1 * np.sin(theta1), -L1 * np.cos(theta1)
        x2, y2 = x1 + L2 * np.sin(theta2), y1 - L2 * np.cos(theta2)
        lines[i].set_data([0, x1, x2], [0, y1, y2])

    ani = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)
    plt.draw()

# --- Arayüz Elemanları ---
ax_text = plt.axes([0.2, 0.05, 0.1, 0.05])
text_box = TextBox(ax_text, 'Sarkaç Sayısı: ', initial="10")

ax_start = plt.axes([0.45, 0.05, 0.2, 0.05])
btn_start = Button(ax_start, 'SİSTEMİ KUR / OYNAT')

def toggle_sim(event):
    global is_running
    if ani is None:
        try:
            n = int(text_box.text)
            run_simulation(n)
            is_running = True
            btn_start.label.set_text('DURAKLAT')
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")
    else:
        is_running = not is_running
        btn_start.label.set_text('DURAKLAT' if is_running else 'DEVAM ET')
    plt.draw()

btn_start.on_clicked(toggle_sim)

# Başlık ve Açıklama
fig.suptitle("Çoklu Çift Sarkaç Simülasyonu", fontsize=16, fontweight='bold')
fig.text(0.5, 0.9, r"Her sarkaç arası $\Delta\theta = 1^\circ$ fark vardır.", 
         ha='center', fontsize=12, style='italic')

plt.show()