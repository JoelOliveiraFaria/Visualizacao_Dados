import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_b4164b73.png'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.palettes import QUALITATIVE

# Carregar dados
df = pd.read_csv("data/gapminder_clean.csv")

# Filtrar para o ano de 2007
df_2007 = df[df['year'] == 2007].copy()

# Configuração do gráfico
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot
# Tamanho da população normalizado para área (s)
scatter = sns.scatterplot(
    data=df_2007,
    x="gdpPercap",
    y="lifeExp",
    hue="continent",
    size="pop",
    sizes=(20, 500),
    palette=QUALITATIVE,
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

# Escalas e eixos
ax.set_xscale("log")
ax.set_xlabel("PIB per capita (USD, escala log)")
ax.set_ylabel("Esperança média de vida (anos)")
ax.set_title("PIB per capita vs Esperança de Vida (2007)")

# Ajuste da legenda
handles, labels = ax.get_legend_handles_labels()
# Remover o título da legenda de tamanho para limpeza visual
ax.legend(handles, labels, title="Continente / População", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)