"""Fase 5 - Estado partilhado do grafo LangGraph."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    """Tudo o que os nós leem/escrevem. `total=False`: os campos vão sendo preenchidos."""
    request: str          # pedido em linguagem natural
    profile: dict         # dicionário de dados (natureza de cada variável)
    code: str             # último código gerado
    audit: dict           # último veredicto do auditor
    feedback: str         # violações do auditor a corrigir (vazio se aprovado)
    error: str            # traceback da última execução (vazio se ok)
    iterations: int       # nº de gerações já feitas
    max_iterations: int   # limite do ciclo (controlo de erros)
    figure_path: str | None
    status: str           # running | success | failed
    # `operator.add` faz o LangGraph ACUMULAR as listas em vez de as substituir:
    # fica o registo completo de todas as iterações (evidência para o relatório).
    history: Annotated[list, operator.add]
