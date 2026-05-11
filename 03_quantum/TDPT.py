import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore")

# 1. SİMÜLASYON PARAMETRELERİ
L0    = 1.0    # Başlangıç kuyu genişliği
L_end = 10     # Bitiş kuyu genişliği
v_fast = 4.0   # Hızlı (irreversible) genişleme hızı
v_slow = 0.005 # Yavaş (adyabatik) genişleme hızı
N_bas  = 12    # Süperpozisyon için durum sayısı
hbar   = 1.0
m      = 1.0

T_fast = (L_end - L0) / v_fast
T_slow = (L_end - L0) / v_slow
frames = 500

t_fast = np.linspace(0, T_fast, frames)
t_slow = np.linspace(0, T_slow, frames)

# ==========================================
# 2. BAĞLANMA MATRİSİ — DOĞRU FORMÜL
# ==========================================
def M_nm(n, m, L):
    if n == m:
        return 0.0
    if (n + m) % 2 == 0:   # aynı parite → integral = 0
        return 0.0
    # İşaret: (-1)^{n+m} = -1 (çünkü n+m tek)
    return -(2.0 * n * m) / (L * (n**2 - m**2))

def build_ODE(v_exp):
    def rhs(t, c):
        L = L0 + v_exp * t
        dc = np.zeros(N_bas, dtype=complex)
        E = np.array([(n+1)**2 * np.pi**2 * hbar**2 / (2*m*L**2)
                      for n in range(N_bas)])
        for i in range(N_bas):
            # Schrödinger fazı
            dc[i] = -1j * (E[i] / hbar) * c[i]
            # Duvar hareketi coupling (dL/dt = v_exp > 0)
            for j in range(N_bas):
                if j != i:
                    Mij = M_nm(i+1, j+1, L)
                    if Mij != 0.0:
                        dc[i] -= v_exp * Mij * c[j]
        return dc
    return rhs

# ==========================================
# 3. ODE ÇÖZÜMÜ
# ==========================================
c0 = np.zeros(N_bas, dtype=complex)
c0[0] = 1.0 + 0j   # Başlangıç: tamamen ground state

print("Simulation Begins...")
sol_fast = solve_ivp(
    build_ODE(v_fast), [0, T_fast], c0, t_eval=t_fast,
    method='DOP853', atol=1e-9, rtol=1e-9
)

sol_slow = solve_ivp(
    build_ODE(v_slow), [0, T_slow], c0, t_eval=t_slow,
    method='DOP853', atol=1e-9, rtol=1e-9
)

c_fast = sol_fast.y   
c_slow = sol_slow.y

# Norm kontrolü
norm_fast_end = np.sum(np.abs(c_fast[:, -1])**2)
norm_slow_end = np.sum(np.abs(c_slow[:, -1])**2)

# ==========================================
# 4. GÖRSEL KURULUM
# ==========================================
x_res = 600

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#0d0f1a')
fig.suptitle("Adiabatic Infinite Square Well - Reversible/Irreversible Processes",
             color='white', fontsize=14, fontweight='bold', y=0.98)

ax_wave_slow, ax_wave_fast = axes[0]
ax_pop_slow,  ax_pop_fast  = axes[1]

COLORS = {
    'slow_wave': '#4fc3f7',
    'fast_wave': '#ff7043',
    'fill_slow': '#1a5276',
    'fill_fast': '#7b2a1c',
    'energy':    '#78909c',
    'exp_slow':  '#4fc3f7',
    'exp_fast':  '#ff7043',
    'wall':      '#eceff1',
    'bg':        '#0d0f1a',
    'grid':      '#1e2030',
    'text':      '#ffffff'
}

n_show = 7

def setup_ax(ax, title, wave_color):
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(-0.1, L_end + 0.3)
    ax.set_ylim(-0.5, 22)
    ax.set_xlabel("Position x", color='#90a4ae', fontsize=10)
    ax.set_ylabel("Energy E", color='#90a4ae', fontsize=10)
    ax.set_title(title, color='white', fontsize=11, pad=6)
    ax.tick_params(colors='#607d8b')
    for sp in ax.spines.values():
        sp.set_color(COLORS['grid'])
    ax.grid(True, color=COLORS['grid'], lw=0.5, alpha=0.6)
    ax.axvline(0, color=COLORS['wall'], lw=3)

setup_ax(ax_wave_slow, f"Reversible (v={v_slow})", COLORS['slow_wave'])
setup_ax(ax_wave_fast, f"Irreversible (v={v_fast})", COLORS['fast_wave'])

for ax in (ax_pop_slow, ax_pop_fast):
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(-0.5, n_show + 0.5)
    # Metinler sığsın diye y limitini 1.05'ten 1.15'e çıkardık
    ax.set_ylim(0, 1.15) 
    ax.tick_params(colors='#607d8b')
    for sp in ax.spines.values():
        sp.set_color(COLORS['grid'])
    ax.grid(True, color=COLORS['grid'], lw=0.5, alpha=0.4, axis='y')
    ax.set_xlabel("Enerji seviyesi n", color='#90a4ae', fontsize=10)
    ax.set_ylabel("$|c_n|²$", color='#90a4ae', fontsize=10)

ax_pop_slow.set_title("Probabilities of Existence at Energy Levels (Reversible)", color='white', fontsize=11)
ax_pop_fast.set_title("Probabilities of Existence at Energy Levels (Irreversible)", color='white', fontsize=11)

wall_slow = ax_wave_slow.axvline(L0, color=COLORS['wall'], lw=3)
wall_fast = ax_wave_fast.axvline(L0, color=COLORS['fast_wave'], lw=2, linestyle='--')

E_lines_slow = [ax_wave_slow.plot([], [], '--', color=COLORS['energy'], lw=1.5, alpha=1.0)[0] for _ in range(n_show)]
E_lines_fast = [ax_wave_fast.plot([], [], '--', color=COLORS['energy'], lw=1.5, alpha=1.0)[0] for _ in range(n_show)]

line_exp_slow, = ax_wave_slow.plot([], [], '-', color=COLORS['exp_slow'], lw=2, label='⟨E⟩')
line_exp_fast, = ax_wave_fast.plot([], [], '-', color=COLORS['exp_fast'], lw=2, label='⟨E⟩')

fill_slow_obj = None
fill_fast_obj = None

bar_slow = ax_pop_slow.bar(range(1, n_show+1), np.zeros(n_show),
                            color=COLORS['slow_wave'], alpha=0.85, width=0.5)
bar_fast = ax_pop_fast.bar(range(1, n_show+1), np.zeros(n_show),
                            color=COLORS['fast_wave'], alpha=0.85, width=0.5)

# YENİ EKLENEN KISIM: Barların üzerine gelecek olan metin objeleri oluşturuluyor
text_slow = [ax_pop_slow.text(i+1, 0, "", ha='center', va='bottom', color=COLORS['text'], fontsize=10, fontweight='bold') for i in range(n_show)]
text_fast = [ax_pop_fast.text(i+1, 0, "", ha='center', va='bottom', color=COLORS['text'], fontsize=10, fontweight='bold') for i in range(n_show)]

txt_norm_fast = ax_wave_fast.text(
    0.98, 0.96, "", transform=ax_wave_fast.transAxes,
    ha='right', va='top', color='#ef5350', fontsize=9
)

wave_scale = 2.5

# ==========================================
# 5. ANİMASYON
# ==========================================
def animate(frame):
    global fill_slow_obj, fill_fast_obj

    # --- Adyabatik frame ---
    t_s = t_slow[frame]
    L_s = L0 + v_slow * t_s
    c_s = c_slow[:, frame]
    x_s = np.linspace(0, L_s, x_res)

    phi_s = np.array([np.sqrt(2/L_s) * np.sin((n+1)*np.pi*x_s/L_s)
                      for n in range(N_bas)])
    E_s   = np.array([(n+1)**2 * np.pi**2 / (2*L_s**2) for n in range(N_bas)])
    pop_s = np.abs(c_s)**2

    psi_s = sum(c_s[n] * phi_s[n] for n in range(N_bas))
    E_exp_s = float(np.real(np.sum(pop_s * E_s)))

    # --- Non-adyabatik frame ---
    t_f = t_fast[frame]
    L_f = L0 + v_fast * t_f
    c_f = c_fast[:, frame]
    x_f = np.linspace(0, L_f, x_res)

    phi_f = np.array([np.sqrt(2/L_f) * np.sin((n+1)*np.pi*x_f/L_f)
                      for n in range(N_bas)])
    E_f   = np.array([(n+1)**2 * np.pi**2 / (2*L_f**2) for n in range(N_bas)])
    pop_f = np.abs(c_f)**2

    psi_f = sum(c_f[n] * phi_f[n] for n in range(N_bas))
    E_exp_f = float(np.real(np.sum(pop_f * E_f)))

    # --- Güncelleme ---
    wall_slow.set_xdata([L_s, L_s])
    wall_fast.set_xdata([L_f, L_f])

    for k in range(n_show):
        E_lines_slow[k].set_data([0, L_s], [E_s[k], E_s[k]])
        E_lines_fast[k].set_data([0, L_f], [E_f[k], E_f[k]])

    line_exp_slow.set_data([0, L_s], [E_exp_s, E_exp_s])
    line_exp_fast.set_data([0, L_f], [E_exp_f, E_exp_f])

    if fill_slow_obj is not None: fill_slow_obj.remove()
    if fill_fast_obj is not None: fill_fast_obj.remove()

    density_s = np.abs(psi_s)**2
    density_f = np.abs(psi_f)**2

    fill_slow_obj = ax_wave_slow.fill_between(
        x_s, E_exp_s, E_exp_s + wave_scale * density_s,
        color=COLORS['slow_wave'], alpha=0.55, linewidth=0
    )
    fill_fast_obj = ax_wave_fast.fill_between(
        x_f, E_exp_f, E_exp_f + wave_scale * density_f,
        color=COLORS['fast_wave'], alpha=0.55, linewidth=0
    )

    # Popülasyon barları ve üzerlerindeki metinlerin anlık güncellenmesi
    pop_s_show = pop_s[:n_show]
    pop_f_show = pop_f[:n_show]
    
    for i, (bar, p) in enumerate(zip(bar_slow, pop_s_show)):
        bar.set_height(p)
        text_slow[i].set_text(f"{p:.3f}") # 3 basamak hassasiyet
        text_slow[i].set_y(p + 0.02)      # Yazıyı barın hafifçe üstüne yerleştir

    for i, (bar, p) in enumerate(zip(bar_fast, pop_f_show)):
        bar.set_height(p)
        text_fast[i].set_text(f"{p:.3f}")
        text_fast[i].set_y(p + 0.02)

    # Norm kontrolü (sayısal hata takibi)
    norm_f = float(np.sum(pop_f))
    if abs(norm_f - 1.0) > 0.01:
        txt_norm_fast.set_text(f"Norm: {norm_f:.4f} !")
    else:
        txt_norm_fast.set_text("")

    return (wall_slow, wall_fast, line_exp_slow, line_exp_fast,
            *E_lines_slow, *E_lines_fast, txt_norm_fast,
            *bar_slow, *bar_fast, *text_slow, *text_fast)

# Butonlar için figürün altında ufak bir boşluk (0.08) bırakıyoruz
plt.tight_layout(rect=[0, 0.08, 1, 0.97])

ani = animation.FuncAnimation(
    fig, animate, frames=frames, interval=40, blit=False
)

# ==========================================
# 6. KONTROL BUTONLARI (PLAY/PAUSE & REPLAY)
# ==========================================
ax_play = plt.axes([0.42, 0.02, 0.07, 0.04])
ax_replay = plt.axes([0.51, 0.02, 0.07, 0.04])

btn_play = Button(ax_play, 'Pause', color=COLORS['grid'], hovercolor='#37474f')
btn_replay = Button(ax_replay, 'Replay', color=COLORS['grid'], hovercolor='#37474f')

btn_play.label.set_color('white')
btn_replay.label.set_color('white')

is_playing = [True]

def toggle_play(event):
    if is_playing[0]:
        ani.pause()
        btn_play.label.set_text('Play')
    else:
        ani.resume()
        btn_play.label.set_text('Pause')
    is_playing[0] = not is_playing[0]
    fig.canvas.draw_idle()

def restart_anim(event):
    ani.frame_seq = ani.new_frame_seq()
    if not is_playing[0]:
        ani.resume()
        btn_play.label.set_text('Pause')
        is_playing[0] = True
    fig.canvas.draw_idle()

btn_play.on_clicked(toggle_play)
btn_replay.on_clicked(restart_anim)

plt.show()