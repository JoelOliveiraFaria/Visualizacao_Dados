"""
Regra de ouro: o TIPO de paleta tem de corresponder à NATUREZA da variável.

  Variável                        Paleta        Exemplo aqui
  ------------------------------  ------------  -------------------
  categórica nominal (sem ordem)  qualitativa   continente
  contínua / ordenada             sequencial    ano
  contínua com centro com sentido divergente    desvio à mediana

O nó de auditoria (Fase 4) verifica exatamente esta correspondência.
"""
from matplotlib.colors import TwoSlopeNorm

# --- Qualitativa: Okabe-Ito ---
# Cores bem distinguíveis entre si e seguras para daltonismo.
# Cada categoria tem UMA cor fixa em todas as figuras (consistência).
QUALITATIVE = {
    "Africa": "#E69F00",    # laranja
    "Americas": "#009E73",  # verde
    "Asia": "#CC79A7",      # rosa
    "Europe": "#0072B2",    # azul
    "Oceania": "#D55E00",   # vermelhão
}

# --- Formas por continente ---
# Codificação redundante (cor + forma): a figura continua legível em
# preto e branco ou por quem não distingue certas cores.
MARKERS = {"Africa": "o", "Americas": "s", "Asia": "^", "Europe": "D", "Oceania": "P"}

# --- Sequencial ---
# viridis é perceptualmente uniforme (passos iguais nos dados = passos iguais
# na perceção) e legível por daltónicos. Usar para variáveis contínuas/ordenadas.
SEQUENTIAL = "viridis"

# --- Divergente ---
# Duas cores a divergir de um centro neutro (claro). SÓ usar quando existe um
# valor central com significado (zero, mediana, média...).
DIVERGING = "RdBu"


def diverging_norm(vmin: float, vmax: float, center: float = 0.0) -> TwoSlopeNorm:
    """Normalização que fixa o centro da escala divergente num valor dado.

    Sem isto, o branco ficaria no meio do intervalo [vmin, vmax] e não no
    valor neutro real, enganando o leitor.
    """
    return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
