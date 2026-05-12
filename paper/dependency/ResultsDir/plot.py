import matplotlib.pyplot as plt
import numpy as np

# Load your data
data_N = np.loadtxt('N_massgain.txt')
data_dt = np.loadtxt('dt_massgain.txt')

# Create a figure with 2 rows and 1 column
fig, axs = plt.subplots(1, 2, figsize=(6, 3))

# First Subplot: Number of finite elements
axs[0].plot(np.array(data_N[:, 0]), np.array(data_N[:, 1]), 's-', color='r', linewidth=2)
axs[0].set_xlabel('Number of finite elements', fontsize=12)
axs[0].set_ylabel('Mass Gain at t=50h', fontsize=12)
# axs[0].grid(True, linestyle='--', alpha=0.7)

# Second Subplot: Timestep
axs[1].plot(np.log10(data_dt[:, 0]), np.log10(data_dt[:, 1]), '^-', color='b', linewidth=2)
axs[1].set_xlabel(r'$\log{\Delta t}$', fontsize=12)
axs[1].set_ylabel('log(Mass Gain at t=50h)', fontsize=12)
# axs[1].grid(True, linestyle='--', alpha=0.7)

# Adjust layout to prevent label overlap
plt.tight_layout()

# Save the figure
plt.savefig('Figure_appendix_dependency.png', dpi=600)

plt.show()