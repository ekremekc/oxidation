import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# plt.rcParams.update({'font.size': 22})


## Oxidation thickness
plt.figure(figsize=(4, 3))

thickness_N_coated = np.loadtxt('fem_data/thickness_N_coated_extrapolation.txt')
thickness_P_coated = np.loadtxt('fem_data/thickness_P_coated_extrapolation.txt')

plt.plot(thickness_N_coated[:, 0], thickness_N_coated[:, 1], '-', color='b', linewidth=2, label='N-Type')
plt.plot(thickness_P_coated[:, 0], thickness_P_coated[:, 1], '-', color='r', linewidth=2, label='P-Type')

plt.xlabel('Time (h)', fontsize=12)
plt.ylabel('Oxygen penetration depth [um]', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
plt.tight_layout()

# Save and show the plot
plt.savefig('plots/oxidation_extrapolation.png', dpi=600)
plt.show()
plt.close()
print("Plot successfully generated and saved as 'oxidation_extrapolation.png'.")