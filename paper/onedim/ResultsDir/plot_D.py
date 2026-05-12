import numpy as np
import matplotlib.pyplot as plt

# --- Define Parameters ---
time = np.linspace(1, 1000, 500)  # Start from 1 to avoid division by zero if k is negative

# N-Type
D0_N = 3e-5 * 3600
k_N = -0.005
D_N = D0_N * np.exp(time * k_N)

# P-Type
D0_P = 9e-6 * 3600
k_P = -0.015
D_P = D0_P * np.exp(time * k_P)

# CrSi
D0_coating = 2e-7 * 3600
k_coating = -0.003
D_coating = D0_coating * np.exp(time * k_coating)

# --- Plotting ---
plt.figure(figsize=(6, 4))

plt.plot(time, np.log10(D_N), label=f'N-Type ($D_0$={D0_N:.3f}, $k$={k_N})', color='blue', linewidth=2)
plt.plot(time, np.log10(D_P), label=f'P-Type ($D_0$={D0_P:.3f}, $k$={k_P})', color='red', linewidth=2)
plt.plot(time, np.log10(D_coating), label=f'CrSi ($D_0$={D0_coating:.3f}, $k$={k_coating})', color='orange', linewidth=2)

# --- Customization ---
plt.xlabel('Time, t (h)', fontsize=14)
plt.ylabel('log(D) [$\mu m^2/h$]', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Adding the inward-facing ticks we discussed earlier
plt.minorticks_on()
plt.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)

plt.tight_layout()
plt.savefig('plots/transient_D.png', dpi=600)
plt.legend()

plt.savefig('plots/Figure_D.pdf')
plt.show()