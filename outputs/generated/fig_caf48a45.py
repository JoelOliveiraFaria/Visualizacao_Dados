import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_caf48a45.png'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.palettes import QUALITATIVE, MARKERS

# Carregar dados
df = pd.read_csv("data/gapminder_clean.csv")

# Filtrar para o ano de 2007
df_2007 = df[df['year'] == 2007].copy()

# Configuração da figura
plt.figure(figsize=(10, 6))

# Scatter plot:
# Eixo X: gdpPercap (log scale conforme sugestão)
# Eixo Y: lifeExp
# Cor: lifeExp (variável contínua, logo usa-se paleta sequencial 'viridis')
# Tamanho: pop (proporcional à área)
scatter = plt.scatter(
    data=df_2007,
    x='gdpPercap',
    y='lifeExp',
    c='lifeExp',
    s=df_2007['pop'] / 1000000,
    cmap='viridis',
    alpha=0.7,
    edgecolors='w',
    linewidth=0.5
)

# Escalas e eixos
plt.xscale('log')
plt.xlabel('PIB per capita (USD, log scale)')
plt.ylabel('Esperança média de vida (anos)')
plt.title('PIB per capita vs Esperança de Vida (2007)')

# Colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Esperança média de vida (anos)')

# Grid e estilo
plt.grid(True, linestyle='--', alpha=0.6)

# Gravar figura
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)