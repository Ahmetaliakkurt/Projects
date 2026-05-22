import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore")

L0    = 1.0
L_end = 10.0
N_bas = 12
hbar  = 1.0
m     = 1.0

def M_nm(n, m, L):
    if n == m:            return 0.0
    if (n + m) % 2 == 0: return 0.0
    return -(2.0 * n * m) / (L * (n**2 - m**2))

def build_ODE(v_exp):
    def rhs(t, c):
        L  = L0 + v_exp * t
        dc = np.zeros(N_bas, dtype=complex)
        E  = np.array([(n+1)**2 * np.pi**2 * hbar**2 / (2*m*L**2)
                       for n in range(N_bas)])
        for i in range(N_bas):
            dc[i] = -1j * (E[i] / hbar) * c[i]
            for j in range(N_bas):
                if j != i:
                    Mij = M_nm(i+1, j+1, L)
                    if Mij != 0.0:
                        dc[i] -= v_exp * Mij * c[j]
        return dc
    return rhs

v_values = np.logspace(np.log10(0.005), np.log10(100.0), 40)

p_ground_list = []   
p_excited_list = []
print("Adyabatik Sınır Analizi Başlıyor... (Bu işlem birkaç saniye sürebilir)")

c0 = np.zeros(N_bas, dtype=complex)
c0[0] = 1.0 + 0j

for v in v_values:
    T_end = (L_end - L0) / v
    
    sol = solve_ivp(build_ODE(v), [0, T_end], c0, 
                    method='DOP853', atol=1e-8, rtol=1e-8)
    
    c_final = sol.y[:, -1]
    
    p1 = np.abs(c_final[0])**2
    
    p_ground_list.append(p1)
    p_excited_list.append(1.0 - p1)

print("Hesaplama Tamamlandı. Grafik Çiziliyor...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d0f1a')
ax.set_facecolor('#0d0f1a')

ax.plot(v_values, p_excited_list, marker='o', color='#ff7043', 
        linewidth=2, markersize=5, label='Total Excitation Probability ($1 - |c_1|^2$)')

ax.axvspan(0.001, 0.1, color='#4fc3f7', alpha=0.1, label='Adiabatic Regime')
ax.axvspan(0.1, 10.0, color='#ffee58', alpha=0.1, label='Transition Regime')
ax.axvspan(10.0, 200.0, color='#ef5350', alpha=0.1, label='Sudden (Non-Quasistatic) Regime')

ax.set_xscale('log')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(min(v_values), max(v_values))

ax.set_title("Quantum Adiabaticity Boundary: Velocity vs Excitation", color='white', fontsize=14, pad=15)
ax.set_xlabel("Expansion Velocity ($v$) — Log Scale", color='#90a4ae', fontsize=12)
ax.set_ylabel("Excitation Probability (Non-Quasistaticity)", color='#90a4ae', fontsize=12)

ax.tick_params(colors='#607d8b', labelsize=10)
for sp in ax.spines.values(): sp.set_color('#1e2030')
ax.grid(True, color='#1e2030', lw=1, alpha=0.8, which="both", ls="--")

legend = ax.legend(loc='center right', facecolor='#0d0f1a', edgecolor='#1e2030', fontsize=10)
for text in legend.get_texts(): text.set_color('white')

plt.tight_layout()
plt.show()
