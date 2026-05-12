import matplotlib.pyplot as plt
import pandas as pd

# Update these values with your actual results
data = {
    'Case': ['N-type', 'P-type', 'Coated N-Type', 'Coated P-Type'],
    'p': [0.43, 0.34, 0.60, 0.15] 
}

df = pd.DataFrame(data)

# Sort the bars as per best practices
df_sorted = df.sort_values(by='p', ascending=False)

# Plotting
plt.figure(figsize=(6, 4))
bars = plt.bar(df['Case'], df['p'], color='skyblue', edgecolor='navy')

plt.xlabel('Oxidation Case', fontsize=12)
plt.ylabel('Oxygen mass fraction, $p$', fontsize=12)
plt.ylim([0, 0.8])
plt.xticks(rotation=15)
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center')

plt.tight_layout()
plt.savefig('plots/massfraction_p.png', dpi=600)
plt.savefig('plots/Figure_massfraction.pdf')
plt.show()