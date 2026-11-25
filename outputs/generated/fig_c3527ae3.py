import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_c3527ae3.png'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from src.palettes import QUALITATIVE, MARKERS

# Carregar dados
df = pd.read_csv("data/gapminder_clean.csv")

# Filtrar para o ano de 2007
df_2007 = df[df['year'] == 2007].copy()

# Configuração da figura
plt.figure(figsize=(10, 6))

# Definir o centro para a paleta divergente (média global da esperança de vida em 2007)
vcenter = df_2007['lifeExp'].mean()
norm = TwoSlopeNorm(vcenter=vcenter, vmin=df_2007['lifeExp'].min(), vmax=df_2007['lifeExp'].max())

# Scatter plot
# O tamanho s é proporcional à área (população). Escala logarítmica aplicada ao eixo x.
scatter = plt.scatter(
    x=df_2007['gdpPercap'],
    y=df_2007['lifeExp'],
    s=df_2007['pop'] / 1000000,
    c=df_2007['lifeExp'],
    cmap='RdBu',
    norm=norm,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Escalas e eixos
plt.xscale('log')
plt.xlabel('PIB per capita (USD, log scale)')
plt.ylabel('Esperança média de vida (anos)')
plt.title('PIB per capita vs Esperança de Vida (2007)')

# Barra de cores com o centro definido
cbar = plt.colorbar(scatter)
cbar.set_label('Esperança média de vida (anos)')

# Ajustes finais
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)