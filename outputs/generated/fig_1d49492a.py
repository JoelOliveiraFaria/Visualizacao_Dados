import matplotlib
matplotlib.use('Agg')
OUTPUT_PATH = 'C:\\Users\\joelo\\Documents\\GitHub\\Visualizacao_Dados\\outputs\\figures\\fig_1d49492a.png'
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("data/gapminder_clean.csv"); d = df[df.year == 2007]
fig, ax = plt.subplots()
ax.scatter(d.gdpPercap, d.lifeExp, c=d.lifeExp, cmap="viridis")
ax.set_xscale("log")
plt.savefig(OUTPUT_PATH)