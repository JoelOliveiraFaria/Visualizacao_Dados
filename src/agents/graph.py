"""Fase 5 - Grafo cíclico LangGraph.

        START -> generate -> audit --(aprovado)--> execute --(ok)--> END
                    ^          |                      |
                    |          +--(reprovado)---------+--(erro de execução)
                    +----------------- feedback / traceback ---+

  - generate : LLM escreve (ou reescreve) o código.
  - audit    : valida escalas de cor ANTES de executar (barato e determinístico).
  - execute  : corre o código; se rebentar, o traceback volta ao gerador.
  - O ciclo termina em sucesso, ou em falha controlada ao atingir max_iterations
    (ou se o LLM ficar indisponível).

Executar:  python -m src.agents.graph "o pedido do gráfico"
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from langgraph.graph import END, START, StateGraph

from src.agents.audit import load_profile
from src.agents.nodes import make_nodes
from src.agents.state import GraphState
from src.data_prep import ROOT

LOG_DIR = ROOT / "outputs" / "logs"
DEFAULT_MAX_ITERATIONS = 4


# --- Funções de roteamento (arestas condicionais) --------------------------
def route_after_generate(state: GraphState) -> str:
    """Se o LLM falhou de vez, termina; senão segue para a auditoria."""
    return "end" if state.get("status") == "failed" else "audit"


def route_after_audit(state: GraphState) -> str:
    """Aprovado -> executar. Reprovado -> reescrever (ou desistir no limite)."""
    if state["audit"]["approved"]:
        return "execute"
    return "give_up" if state["iterations"] >= state["max_iterations"] else "generate"


def route_after_execute(state: GraphState) -> str:
    """Sucesso -> fim. Erro -> reescrever (ou desistir no limite)."""
    if state.get("status") == "success":
        return "end"
    return "give_up" if state["iterations"] >= state["max_iterations"] else "generate"


def give_up(state: GraphState) -> dict:
    """Saída controlada: marca falha e guarda a razão, sem lançar exceção."""
    reason = state.get("feedback") or state.get("error") or "limite de iterações atingido"
    return {"status": "failed", "error": f"Desistiu após {state['iterations']} iterações. Último problema: {reason}"}


def build_graph(llm, judge_llm=None):
    """Constrói e compila o grafo com o LLM dado."""
    nodes = make_nodes(llm, judge_llm)
    g = StateGraph(GraphState)
    g.add_node("generate", nodes["generate"])
    g.add_node("audit", nodes["audit"])
    g.add_node("execute", nodes["execute"])
    g.add_node("give_up", give_up)

    g.add_edge(START, "generate")
    g.add_conditional_edges("generate", route_after_generate, {"audit": "audit", "end": END})
    g.add_conditional_edges("audit", route_after_audit,
                            {"execute": "execute", "generate": "generate", "give_up": "give_up"})
    g.add_conditional_edges("execute", route_after_execute,
                            {"end": END, "generate": "generate", "give_up": "give_up"})
    g.add_edge("give_up", END)
    return g.compile()


def get_llm(model: str | None = None):
    """Cria o LLM Gemini (chave em GOOGLE_API_KEY, lida do .env)."""
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI
    load_dotenv()
    return ChatGoogleGenerativeAI(model=model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), temperature=0.2,
                                  timeout=int(os.getenv("LLM_TIMEOUT", "60")), max_retries=1)


def run_request(request: str, llm=None, judge_llm=None, max_iterations: int = DEFAULT_MAX_ITERATIONS,
                label: str = "run") -> dict:
    """Corre o grafo para um pedido, grava o log JSON e devolve o estado final."""
    llm = llm or get_llm()
    graph = build_graph(llm, judge_llm)
    final = graph.invoke({"request": request, "profile": load_profile(), "iterations": 0,
                          "max_iterations": max_iterations, "history": [], "status": "running"},
                         config={"recursion_limit": 6 * max_iterations + 10})
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{label}_{datetime.now():%Y%m%d_%H%M%S}.json"
    log.write_text(json.dumps({k: v for k, v in final.items() if k != "profile"},
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    final["log_path"] = str(log)
    return final


def save_diagram(path=None) -> str:
    """Grava o diagrama do grafo em Mermaid (para o relatório). Não precisa de LLM."""
    path = path or (ROOT / "outputs" / "graph.mmd")
    class _Dummy:  # só para compilar o grafo
        def invoke(self, *_a, **_k): raise RuntimeError
    path.write_text(build_graph(_Dummy()).get_graph().draw_mermaid(), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    req = " ".join(sys.argv[1:]) or "Scatter de PIB per capita vs esperança de vida em 2007, cor por continente, tamanho por população."
    out = run_request(req)
    print("estado:", out["status"], "| iterações:", out["iterations"])
    print("figura:", out.get("figure_path"))
    print("log:", out["log_path"])
    if out["status"] != "success":
        print("erro:", out.get("error"))
