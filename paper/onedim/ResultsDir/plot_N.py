import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Experimental data
file_path = 'experimental_data/Mass_Gain_N_P1_Uncoated_2.csv'
df = pd.read_csv(file_path)

# FEM data
data_N = np.loadtxt('fem_data/massGain_N.txt')

data_N_coated = np.loadtxt('fem_data/massGain_N_coated.txt')

plt.figure(figsize=(6, 4))

rho_substrate = 6.021464*1e-8 # mg/micrometer3

# Not coated
pct_o2 = 43/100 # percentage
solubility_N_exp = pct_o2*rho_substrate*1e8 #mg/micrometer3

plt.plot(data_N[:, 0], data_N[:, 1]*solubility_N_exp, '-', color='r', linewidth=2, label=r'$m_{f,u}$')
plt.plot(df['t'], df['N-50'], label=r'$m_{e,u}$', marker='o', linestyle='-')

# Coated
pct_o2_coated = 60/100 # percentage
solubility_N_exp_coated = pct_o2_coated*rho_substrate*1e8 #mg/micrometer3

plt.plot(data_N_coated[:, 0], data_N_coated[:, 1]*solubility_N_exp_coated, '-', color='b', linewidth=2, label=r'$m_{f,c}$')
plt.plot(df['t'], df['Coated N-50'], label=r'$m_{e,c}$', marker='^', linestyle='-')

# Add labels and title
plt.xlabel('Time (h)', fontsize=12)
plt.ylabel('Mass Gain [mg/cm2]', fontsize=12)

# Final touches
# plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save and show the plot
plt.savefig('plots/massgain_N.png', dpi=600)
plt.show()
plt.close()
print("Plot successfully generated and saved as 'massgain_plot.png'.")

## Oxidation thickness
plt.figure(figsize=(6, 4))

thickness_N = np.loadtxt('fem_data/thickness_N.txt')
thickness_N_coated = np.loadtxt('fem_data/thickness_N_coated.txt')

# plt.plot(thickness_N[:, 0], thickness_N[:, 1], '-', color='r', linewidth=2, label='Uncoated')
plt.plot(thickness_N_coated[:, 0], thickness_N_coated[:, 1], '-', color='b', linewidth=2, label='Coated')

plt.xlabel('Time (h)', fontsize=12)
plt.ylabel('Oxygen penetration depth [um]', fontsize=12)

# plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save and show the plot
plt.savefig('plots/oxidation_N.png', dpi=300)
plt.show()
plt.close()
print("Plot successfully generated and saved as 'oxidation_N.png'.")