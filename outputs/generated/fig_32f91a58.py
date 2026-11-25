import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_32f91a58.png'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.palettes import QUALITATIVE

# Carregar dados
df = pd.read_csv("data/gapminder_clean.csv")

# Filtrar para o ano de 2007
df_2007 = df[df['year'] == 2007].copy()

# Configuração da figura
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Plotagem: PIB per capita (x) vs Esperança de vida (y)
# Correção: Uso de paleta qualitativa para variável categórica nominal 'continent'
scatter = sns.scatterplot(
    data=df_2007,
    x='gdpPercap',
    y='lifeExp',
    size='pop',
    sizes=(20, 500),
    hue='continent',
    palette=QUALITATIVE,
    alpha=0.7,
    edgecolor='black'
)

# Escalas e eixos
plt.xscale('log')
plt.xlabel("PIB per capita (USD, log scale)")
plt.ylabel("Esperança média de vida (anos)")
plt.title("PIB per capita vs Esperança de Vida (2007)")

# Ajuste da legenda
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Continente")

# Gravar figura
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)