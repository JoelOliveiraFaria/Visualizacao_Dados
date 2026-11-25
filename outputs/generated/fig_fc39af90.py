import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_fc39af90.png'
import pandas as pd
import matplotlib.pyplot as plt
from src.palettes import QUALITATIVE, MARKERS

# Carregar dados
df = pd.read_csv("data/gapminder_clean.csv")

# Filtrar para o ano de 2007
df_2007 = df[df['year'] == 2007].copy()

# Configuração da figura
plt.figure(figsize=(10, 6))

# Scatter plot: PIB per capita (x) vs Esperança de vida (y)
# População mapeada para cor (sequencial 'viridis') e tamanho (área)
scatter = plt.scatter(
    x=df_2007['gdpPercap'],
    y=df_2007['lifeExp'],
    s=(df_2007['pop'] / 1000000) * 2,  # Escalonamento para visibilidade
    c=df_2007['pop'],
    cmap='viridis',
    alpha=0.7,
    edgecolors='white',
    linewidth=0.5
)

# Escalas logarítmicas conforme dicionário
plt.xscale('log')

# Rótulos e Título
plt.xlabel("PIB per capita (USD, escala log)")
plt.ylabel("Esperança média de vida (anos)")
plt.title("PIB per capita vs Esperança de Vida (2007)")

# Barra de cores
cbar = plt.colorbar(scatter)
cbar.set_label("População total")

# Gravar figura
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)