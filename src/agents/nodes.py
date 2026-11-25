"""Fases 3 e 5 - Nós do grafo: gerador, auditor e executor.

Cada nó é uma função `estado -> atualização parcial do estado`. São criados por
`make_nodes(...)` para receberem o LLM por injeção (facilita testes com LLM falso).
"""
from __future__ import annotations

import json
import re
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src import palettes
from src.agents.audit import audit_color
from src.agents.executor import run_code
from src.agents.state import GraphState

# ---------------------------------------------------------------------------
# Prompt do gerador
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """És um programador especialista em visualização de dados com Python (matplotlib, seaborn, pandas).
Geras UM script Python completo, executável, que produz UMA figura estática a partir do dataset Gapminder.

AMBIENTE (já definido antes do teu código - NÃO redefinir):
- matplotlib já está com backend 'Agg'. A variável OUTPUT_PATH contém o caminho onde gravar a figura.
- Gravar SEMPRE com: plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200). Nunca usar plt.show().
- Ler os dados com: df = pd.read_csv("data/gapminder_clean.csv")  (colunas no dicionário de dados abaixo).
- Podes importar de src.palettes: QUALITATIVE (dict continente->cor), MARKERS (dict continente->forma),
  SEQUENTIAL ('viridis'), DIVERGING ('RdBu'), diverging_norm(vmin, vmax, center).

REGRAS DE COR (Wilke, cap. 4) - o teu código será AUDITADO e rejeitado se as violar:
- variável categórica nominal -> paleta QUALITATIVA (usar .map(QUALITATIVE) ou palette qualitativa);
- variável contínua ou ordenada (ex.: year) -> paleta SEQUENCIAL (cmap='viridis');
- paleta DIVERGENTE só se houver centro com significado e DEVE definir o centro (norm=diverging_norm(...) ou TwoSlopeNorm);
- proibido arco-íris (jet, rainbow, hsv, nipy_spectral, gist_rainbow);
- evitar mais de 8 categorias em paleta qualitativa.

REGRAS DE ESCALA E MAPEAMENTO (Wilke, caps. 2-3):
- se uma variável quantitativa tiver "suggest_log_scale": true, usar escala logarítmica no eixo respetivo;
- tamanho de marcador proporcional à ÁREA (parâmetro s), nunca ao raio;
- título, rótulos dos eixos com unidades e legenda/colorbar sempre presentes;
- filtrar um único ano quando o pedido não disser o contrário (usar 2007).

RESPOSTA: apenas um bloco ```python ... ``` com o código completo, sem explicações."""


def _slim_profile(profile: dict) -> str:
    """Versão compacta do dicionário de dados para o prompt (poupa tokens)."""
    keys = ("nature", "description", "cardinality", "min", "max", "skew", "suggest_log_scale")
    return json.dumps({c: {k: v[k] for k in keys if k in v} for c, v in profile.items()},
                      ensure_ascii=False, indent=1)


def build_user_prompt(state: GraphState) -> str:
    """Monta o prompt do utilizador; em reescritas inclui o código anterior e o problema."""
    parts = [f"DICIONÁRIO DE DADOS:\n{_slim_profile(state['profile'])}",
             f"PEDIDO: {state['request']}"]
    if state.get("code") and (state.get("feedback") or state.get("error")):
        parts.append(f"O teu código anterior:\n```python\n{state['code']}\n```")
        if state.get("feedback"):
            parts.append("FOI REJEITADO PELO AUDITOR DE COR. Corrige estas violações:\n" + state["feedback"])
        if state.get("error"):
            parts.append("FALHOU NA EXECUÇÃO. Corrige este erro:\n" + state["error"])
        parts.append("Reescreve o script completo.")
    return "\n\n".join(parts)


def extract_code(text: str) -> str:
    """Extrai o bloco ```python``` da resposta (ou o texto todo se não houver bloco)."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _content_to_text(content) -> str:
    """Alguns modelos devolvem `content` como lista de blocos; normaliza para str."""
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


# ---------------------------------------------------------------------------
# Fábrica de nós
# ---------------------------------------------------------------------------
def make_nodes(llm, judge_llm=None, max_iterations: int = 4) -> dict[str, Callable]:
    """Cria os três nós ligados ao LLM dado.

    llm        : modelo que gera o código.
    judge_llm  : modelo opcional para a camada LLM-as-judge do auditor.
    """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
    def _call_llm(messages):
        # Retry com backoff exponencial: cobre falhas transitórias/rate limit da API.
        return llm.invoke(messages)

    def generate(state: GraphState) -> dict:
        """NÓ GERADOR: pede (ou reescreve) o código ao LLM."""
        n = state.get("iterations", 0) + 1
        try:
            resp = _call_llm([SystemMessage(content=SYSTEM_PROMPT),
                              HumanMessage(content=build_user_prompt(state))])
        except Exception as e:  # esgotadas as tentativas: sai do ciclo de forma controlada
            return {"iterations": n, "status": "failed", "error": f"Falha do LLM: {type(e).__name__}: {e}"}
        code = extract_code(_content_to_text(resp.content))
        # Limpa o feedback anterior: já foi consumido por esta geração.
        return {"code": code, "iterations": n, "feedback": "", "error": "", "status": "running"}

    def audit(state: GraphState) -> dict:
        """NÓ AUDITOR: valida as escalas de cor do código (estático + juiz LLM opcional)."""
        result = audit_color(state["code"], state["profile"], judge_llm)
        return {"audit": result.to_dict(),
                "feedback": "" if result.approved else result.feedback(),
                "history": [{"iteration": state["iterations"], "stage": "audit",
                             "approved": result.approved, "findings": result.to_dict()["findings"],
                             "code": state["code"]}]}

    def execute(state: GraphState) -> dict:
        """NÓ EXECUTOR: corre o código num subprocesso e devolve figura ou traceback."""
        res = run_code(state["code"], name="fig")
        update = {"error": res.error, "figure_path": res.figure_path,
                  "history": [{"iteration": state["iterations"], "stage": "execute", "ok": res.ok,
                               "error": res.error, "script": res.script_path}]}
        if res.ok:
            update["status"] = "success"
        return update

    return {"generate": generate, "audit": audit, "execute": execute}
