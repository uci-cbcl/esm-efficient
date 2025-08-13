import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


df = pd.read_csv(snakemake.input['predictions'])

plt.figure(figsize=(4, 4), dpi=300)
plt.plot([40, 100], [40, 100], color='black', linestyle='--', alpha=0.5)
sns.scatterplot(data=df, x='Predicted Melting Point', y='True Melting Point', alpha=0.5)
stats = spearmanr(df['Predicted Melting Point'], df['True Melting Point'])
plt.text(0.05, 0.9, r'$\rho$' + f': {stats.correlation:.2f}', transform=plt.gca().transAxes)
sns.despine()
plt.savefig(snakemake.output['scatter_test'], bbox_inches='tight', dpi=300, transparent=True)