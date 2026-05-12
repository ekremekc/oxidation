import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## Oxidation thickness
plt.figure(figsize=(4, 3))

thickness_P_coated0 = np.loadtxt('fem_data/massGain_P_coated_thickness_2.5.txt')
thickness_P_coated2 = np.loadtxt('fem_data/massGain_P_coated_thickness_0.25.txt')
thickness_P_coated3 = np.loadtxt('fem_data/massGain_P_coated_thickness_0.1.txt')
thickness_P_coated4 = np.loadtxt('fem_data/massGain_P_coated_thickness_0.05.txt')

plt.plot(thickness_P_coated0[:, 0], thickness_P_coated0[:, 1], '-', linewidth=2, label='2.5')
plt.plot(thickness_P_coated2[:, 0], thickness_P_coated2[:, 1], '-', linewidth=2, label='0.25')
plt.plot(thickness_P_coated3[:, 0], thickness_P_coated3[:, 1], '-', linewidth=2, label='0.1')
plt.plot(thickness_P_coated4[:, 0], thickness_P_coated4[:, 1], '-', linewidth=2, label='0.05')

plt.xlabel('Time (h)')
plt.ylabel('Mass Gain [mg/cm2]')
plt.ylim([0.0, 1.2])

# plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save and show the plot
plt.savefig('plots/coatingThickness_P.png', dpi=600)
plt.show()
plt.close()
print("Plot successfully generated and saved as 'coatingThickness_P.png'.")