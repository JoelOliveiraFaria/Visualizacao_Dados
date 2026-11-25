"""
Este módulo faz três coisas:
  1. Lê o ficheiro bruto (data/gapminder.tsv) e valida-o.
  2. Guarda uma versão limpa (data/gapminder_clean.csv).
  3. Gera um "dicionário de dados" (data/variable_profile.json) que descreve
     a NATUREZA de cada variável. Este perfil é a peça-chave para o nó de
     auditoria de cor: é ele que diz se uma variável é contínua ou categórica.

Executar:  python -m src.data_prep
"""
import json
from pathlib import Path

import pandas as pd

# --- Caminhos (relativos à raiz do projeto, funcionam em qualquer máquina) ---
ROOT = Path(__file__).resolve().parent.parent      # pasta do projeto
RAW = ROOT / "data" / "gapminder.tsv"              # dados brutos (nunca alterados)
CLEAN = ROOT / "data" / "gapminder_clean.csv"      # dados tratados
PROFILE = ROOT / "data" / "variable_profile.json"  # dicionário de dados

# --- Dicionário 1: natureza semântica de cada variável ---
# Segue a taxonomia de Wilke (cap. 2/4): decide que tipo de escala de cor é
# adequada (qualitativa para nominal, sequencial para contínua, etc.).
NATURE = {
    "country": "categorical_nominal",       # 142 países, sem ordem
    "continent": "categorical_nominal",     # 5 continentes, sem ordem
    "year": "ordinal_temporal",             # ordenada (1952..2007, de 5 em 5)
    "lifeExp": "quantitative_continuous",   # anos de vida
    "pop": "quantitative_continuous",       # habitantes
    "gdpPercap": "quantitative_continuous", # USD por pessoa
}

# --- Dicionário 2: descrição legível de cada coluna (para relatório/LLM) ---
DESCRIPTION = {
    "country": "País",
    "continent": "Continente",
    "year": "Ano (de 5 em 5 anos, 1952-2007)",
    "lifeExp": "Esperança média de vida à nascença (anos)",
    "pop": "População total",
    "gdpPercap": "PIB per capita (USD, ajustado à inflação)",
}


def load_clean() -> pd.DataFrame:
    """Lê o TSV bruto, valida e devolve um DataFrame limpo.

    Validações: colunas esperadas presentes, sem duplicados, sem nulos.
    Conversões: tipos corretos (int para ano/população; category para texto).
    """
    df = pd.read_csv(RAW, sep="\t")  # o ficheiro é separado por tabs

    # Falha cedo e com mensagem clara se faltar alguma coluna esperada.
    missing = set(NATURE) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas em falta: {missing}")

    df = df.drop_duplicates().dropna()
    df["year"] = df["year"].astype(int)
    df["pop"] = df["pop"].astype("int64")
    # 'category' poupa memória e marca explicitamente que são variáveis nominais.
    for c in ("country", "continent"):
        df[c] = df[c].astype("category")
    return df.reset_index(drop=True)


def build_profile(df: pd.DataFrame) -> dict:
    """Constrói o perfil (dicionário de dados) de cada variável.

    Para cada coluna devolve: natureza, descrição, tipo, cardinalidade e nº de
    nulos. Para quantitativas acrescenta min/max/assimetria e uma sugestão de
    escala logarítmica; para categóricas, uma amostra dos níveis.
    """
    prof = {}
    for col, nature in NATURE.items():
        s = df[col]
        info = {
            "nature": nature,
            "description": DESCRIPTION[col],
            "dtype": str(s.dtype),
            "cardinality": int(s.nunique()),  # nº de valores distintos
            "n_missing": int(s.isna().sum()),
        }
        # Intervalo de valores (só faz sentido para variáveis ordenáveis).
        if nature.startswith("quantitative") or nature == "ordinal_temporal":
            info.update(min=float(s.min()), max=float(s.max()))
        if nature.startswith("quantitative"):
            # Assimetria (skewness): > 1 significa cauda longa à direita.
            info["skew"] = round(float(s.skew()), 2)
            # Regra prática: assimetria forte e valores positivos -> escala log
            # (Wilke cap. 3). Aqui isto devolve True para pop e gdpPercap.
            info["suggest_log_scale"] = bool(s.skew() > 1 and s.min() > 0)
        else:
            info["levels_sample"] = [str(v) for v in s.unique()[:10]]
        prof[col] = info
    return prof


def main() -> None:
    """Ponto de entrada: dados brutos -> CSV limpo + perfil JSON."""
    df = load_clean()
    df.to_csv(CLEAN, index=False)
    PROFILE.write_text(json.dumps(build_profile(df), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(df)} linhas -> {CLEAN.name}, {PROFILE.name}")


if __name__ == "__main__":
    main()
