import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_3f0d5d92.png'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.palettes import QUALITATIVE

# Carregar dados
df = pd.read_csv("data/gapminder_clean.csv")

# Filtrar para o ano de 2007
df_2007 = df[df['year'] == 2007].copy()

# Configuração do gráfico
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

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
    linewidth=0.5
)

# Escalas e eixos
plt.xscale("log")
plt.xlabel("PIB per capita (USD, escala log)")
plt.ylabel("Esperança média de vida (anos)")
plt.title("Relação entre PIB per capita e Esperança de Vida (2007)")

# Ajuste da legenda
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

# Gravar figura
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)