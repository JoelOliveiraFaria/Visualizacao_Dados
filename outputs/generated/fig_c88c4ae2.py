import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_c88c4ae2.png'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.palettes import QUALITATIVE

# Carregar dados
df = pd.read_csv("data/gapminder_clean.csv")

# Filtrar para o ano de 2007
df_2007 = df[df['year'] == 2007].copy()

# Configurar estilo
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Criar scatter plot
# Tamanho da população normalizado para melhor visualização (s)
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

# Configurações de escala
ax.set_xscale("log")
ax.set_xlabel("PIB per capita (USD, log scale)")
ax.set_ylabel("Esperança média de vida (anos)")
ax.set_title("PIB per capita vs Esperança de Vida (2007)")

# Ajustar legenda
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

# Gravar figura
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)