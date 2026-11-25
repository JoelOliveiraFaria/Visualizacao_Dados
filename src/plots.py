"""Fase 2 - Figuras base F1-F6 (referência escrita à mão, sem agentes).

Servem de "padrão de qualidade": mais tarde o agente LangGraph tem de gerar
gráficos equivalentes e o auditor valida-os contra estas mesmas regras.
Todas as figuras são geradas por código (nenhuma edição manual).

Executar:  python -m src.plots
"""
import matplotlib
matplotlib.use("Agg")  # backend sem janela: só grava ficheiros (reprodutível)
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

from src.data_prep import CLEAN, ROOT
from src.palettes import DIVERGING, MARKERS, QUALITATIVE, SEQUENTIAL, diverging_norm

OUT = ROOT / "outputs" / "figures"  # onde as figuras são gravadas
YEAR = 2007                          # ano de referência (o mais recente)

# Estilo global: remove contornos superior/direito (menos "tinta" inútil,
# Wilke cap. 19) e fixa resolução de gravação.
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


# ----------------------------------------------------------------------------
# Funções auxiliares (partilhadas por todas as figuras)
# ----------------------------------------------------------------------------
def _data() -> pd.DataFrame:
    """Carrega o CSV limpo produzido por data_prep.py."""
    return pd.read_csv(CLEAN)


def _size(pop, max_area=900):
    """Converte população em tamanho de marcador.

    O argumento `s` do scatter é a ÁREA do marcador. Ao torná-lo proporcional
    à população, a área (e não o raio) representa a quantidade, como
    recomenda Wilke. O maior país fica com `max_area`.
    """
    return max_area * pop / pop.max()


def _money_log(ax, ticks=(500, 1000, 2000, 5000, 10000, 20000, 50000)):
    """Põe o eixo x em escala logarítmica com rótulos legíveis (1 000, 5 000...)."""
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " ")))
    ax.xaxis.set_minor_formatter(mt.NullFormatter())  # sem rótulos menores


def _save(fig, name):
    """Grava a figura em outputs/figures/ e fecha-a (liberta memória)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _continent_legend(ax, markers=False):
    """Legenda manual de continentes (cor, e opcionalmente forma)."""
    h = [plt.Line2D([], [], ls="", marker=MARKERS[c] if markers else "o", color=col, label=c,
                    markersize=8, markeredgecolor="white") for c, col in QUALITATIVE.items()]
    ax.legend(handles=h, title="Continente", frameon=False, loc="lower right")


# ----------------------------------------------------------------------------
# F1 - F6
# ----------------------------------------------------------------------------
def f1_position_hue_size(d):
    """F1: quatro variáveis num só gráfico.

    x = PIB (posição, log) | y = esperança de vida (posição)
    matiz = continente (qualitativa) | área = população (tamanho)
    Mostra o mapeamento estético de variáveis quantitativas e qualitativas.
    """
    d = d[d.year == YEAR]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(d.gdpPercap, d.lifeExp, s=_size(d["pop"]), c=d.continent.map(QUALITATIVE),
               alpha=0.7, edgecolor="white", linewidth=0.5)  # alpha: lida com sobreposição
    _money_log(ax)
    ax.set(xlabel="PIB per capita (USD, escala log)", ylabel="Esperança de vida (anos)",
           title=f"Riqueza e esperança de vida, {YEAR}")
    _continent_legend(ax)
    ax.text(0.01, 0.02, "Área do círculo ∝ população", transform=ax.transAxes, fontsize=8, color="gray")
    return _save(fig, "F1_scatter_hue_size.png")


def f2_linear_vs_log(d):
    """F2: justificação da escala logarítmica.

    Em escala linear, a maioria dos países fica espremida junto ao eixo y
    (PIB tem assimetria 3.85). Em log, a relação torna-se aproximadamente
    linear e todos os países ficam visíveis. Compara os dois painéis.
    """
    d = d[d.year == YEAR]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)  # mesmo eixo y
    for ax, scale in zip(axes, ("linear", "log")):
        ax.scatter(d.gdpPercap, d.lifeExp, s=30, c=d.continent.map(QUALITATIVE), alpha=0.8,
                   edgecolor="white", linewidth=0.4)
        if scale == "log":
            _money_log(ax, ticks=(500, 1000, 5000, 20000, 50000))  # poucos ticks: painel estreito
        ax.set(xlabel="PIB per capita (USD)", title=f"Escala {scale}")
    axes[0].set_ylabel("Esperança de vida (anos)")
    axes[0].annotate("Maioria dos países\ncomprimida", xy=(1500, 45), xytext=(15000, 45),
                     arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")
    axes[1].text(0.03, 0.95, "Relação aproximadamente linear em log", transform=axes[1].transAxes,
                 fontsize=8, color="gray", va="top")
    _continent_legend(axes[1])
    fig.suptitle("Escala log lineariza a relação (PIB é assimétrico à direita)", y=1.02)
    return _save(fig, "F2_linear_vs_log.png")


def f3_shape_redundant(d):
    """F3: forma + cor para o mesmo atributo (continente).

    Codificação redundante: se a cor falhar (daltonismo, impressão a P&B),
    a forma continua a distinguir os grupos.
    """
    d = d[d.year == YEAR]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c, g in d.groupby("continent", observed=True):  # um scatter por continente
        ax.scatter(g.gdpPercap, g.lifeExp, marker=MARKERS[c], s=55, color=QUALITATIVE[c],
                   edgecolor="white", linewidth=0.5, label=c)
    _money_log(ax)
    ax.set(xlabel="PIB per capita (USD, escala log)", ylabel="Esperança de vida (anos)",
           title=f"Continente codificado por forma e cor, {YEAR}")
    ax.legend(title="Continente", frameon=False, loc="lower right")
    return _save(fig, "F3_shape_hue_redundant.png")


def f4_sequential(d):
    """F4: escala de cor SEQUENCIAL para uma variável ordenada (ano).

    O ano é contínuo/ordenado, logo usa-se viridis (escuro->claro = antigo->
    recente) e uma barra de cor, em vez de uma paleta qualitativa.
    """
    d = d[d.continent == "Europe"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(d.gdpPercap, d.lifeExp, c=d.year, cmap=SEQUENTIAL, s=28, alpha=0.9)
    _money_log(ax)
    ax.set(xlabel="PIB per capita (USD, escala log)", ylabel="Esperança de vida (anos)",
           title="Europa: trajetória 1952-2007 (cor sequencial = ano)")
    fig.colorbar(sc, ax=ax, label="Ano")
    return _save(fig, "F4_sequential_year.png")


def f5_diverging(d):
    """F5: escala de cor DIVERGENTE com centro com significado.

    Cor = esperança de vida menos a mediana mundial (2007). O branco fica
    exatamente na mediana (desvio 0): vermelho = abaixo, azul = acima.
    y = ganho de esperança de vida desde 1952 (linha preta em 0 = sem ganho).
    """
    d = d[d.year == YEAR].copy()
    # Esperança de vida de cada país em 1952, para calcular o ganho.
    base = pd.read_csv(CLEAN).query("year == 1952").set_index("country").lifeExp
    d["gain"] = d.lifeExp - d.country.map(base)
    med = d.lifeExp.median()
    d["dev"] = d.lifeExp - med  # desvio à mediana (centro = 0)
    lim = np.abs(d.dev).max()   # limites simétricos: cores comparáveis dos dois lados
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(d.gdpPercap, d.gain, c=d.dev, cmap=DIVERGING, norm=diverging_norm(-lim, lim, 0),
                    s=40, edgecolor="gray", linewidth=0.3)
    ax.axhline(0, color="black", lw=0.6)
    _money_log(ax)
    ax.set(xlabel="PIB per capita (USD, escala log)", ylabel="Ganho de esperança de vida desde 1952 (anos)",
           title=f"Ganho desde 1952 (cor: desvio à mediana mundial de {med:.1f} anos, {YEAR})")
    fig.colorbar(sc, ax=ax, label="Esperança de vida − mediana (anos)")
    return _save(fig, "F5_diverging_median.png")


def f6_qualitative_palette(d):
    """F6: paleta QUALITATIVA aplicada a uma variável nominal.

    Uma cor por continente, sem sugerir ordem. As linhas horizontais são
    categorias; o pequeno deslocamento vertical (jitter, semente fixa para ser
    reprodutível) evita que os pontos se sobreponham.
    """
    d = d[d.year == YEAR]
    rng = np.random.default_rng(0)  # semente fixa -> figura idêntica em cada execução
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (c, col) in enumerate(QUALITATIVE.items()):
        mask = d.continent == c
        ax.scatter(d[mask].lifeExp, np.full(mask.sum(), i) + rng.uniform(-0.15, 0.15, mask.sum()),
                   color=col, s=30, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(QUALITATIVE)), list(QUALITATIVE))
    ax.set(xlabel="Esperança de vida (anos)", title=f"Paleta qualitativa Okabe-Ito, {YEAR}")
    return _save(fig, "F6_qualitative_palette.png")


def main():
    """Gera as seis figuras a partir do CSV limpo."""
    d = _data()
    for f in (f1_position_hue_size, f2_linear_vs_log, f3_shape_redundant,
              f4_sequential, f5_diverging, f6_qualitative_palette):
        print(f.__name__, "->", f(d).name)


if __name__ == "__main__":
    main()
