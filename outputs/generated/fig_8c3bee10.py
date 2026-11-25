import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_8c3bee10.png'
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("data/gapminder_clean.csv"); d = df[df.year == 2007]
fig, ax = plt.subplots(figsize=(7, 4.5))
sc = ax.scatter(d.gdpPercap, d.lifeExp, c=d.lifeExp, cmap="viridis")
ax.set_xscale("log"); ax.set(xlabel="PIB per capita (USD, log)", ylabel="Esperança de vida (anos)", title="2007")
fig.colorbar(sc, label="Esperança de vida (anos)")
plt.savefig(OUTPUT_PATH, bbox_inches="tight")