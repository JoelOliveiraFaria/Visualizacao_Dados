import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_8c7e7f95.png'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.palettes import QUALITATIVE

# Configuração do ambiente
df = pd.read_csv("data/gapminder_clean.csv")
data_2007 = df[df['year'] == 2007].copy()

# Configuração do estilo
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Mapeamento de tamanho: normalizar população para área (s)
# Usamos uma escala para que os pontos não fiquem muito pequenos ou grandes
sizes = (data_2007['pop'] / data_2007['pop'].max()) * 1000

# Plotagem
scatter = sns.scatterplot(
    data=data_2007,
    x='gdpPercap',
    y='lifeExp',
    hue='continent',
    size=sizes,
    palette=QUALITATIVE,
    sizes=(20, 500),
    alpha=0.7,
    edgecolor='black',
    linewidth=0.5,
    ax=ax
)

# Escalas e eixos
ax.set_xscale('log')
ax.set_xlabel('PIB per capita (USD, log scale)')
ax.set_ylabel('Esperança média de vida (anos)')
ax.set_title('PIB per capita vs Esperança de Vida (2007)')

# Ajuste da legenda
handles, labels = scatter.get_legend_handles_labels()
# Remove o handle do tamanho da população da legenda principal para limpar
ax.legend(handles[:6], labels[:6], title="Continente", loc='lower right')

plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)