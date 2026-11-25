"""Testes do grafo (Fase 5) com LLM falso: sem rede, sem custos."""
from types import SimpleNamespace

from src.agents.graph import build_graph, run_request, route_after_audit
from src.agents.audit import load_profile
from src.agents.nodes import extract_code

GOOD = '''import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("data/gapminder_clean.csv"); d = df[df.year == 2007]
fig, ax = plt.subplots()
ax.scatter(d.gdpPercap, d.lifeExp, c=d.lifeExp, cmap="viridis")
ax.set_xscale("log")
plt.savefig(OUTPUT_PATH)
'''
BAD_COLOR = GOOD.replace('cmap="viridis"', 'cmap="tab10"')           # R1: contínua + qualitativa
BAD_RUNTIME = GOOD.replace("d.gdpPercap", "d.nao_existe")             # AttributeError
INFINITE = "while True:\n    pass\n"


class FakeLLM:
    """Devolve respostas pré-definidas, por ordem; regista os prompts recebidos."""
    def __init__(self, *codes):
        self.codes, self.prompts = list(codes), []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        return SimpleNamespace(content=f"```python\n{code}```")


def test_extract_code():
    assert extract_code("texto\n```python\nx=1\n```\nfim") == "x=1"
    assert extract_code("x=1") == "x=1"


def test_valid_first_try():
    out = run_request("teste", FakeLLM(GOOD), label="t_ok")
    assert out["status"] == "success" and out["iterations"] == 1
    assert out["figure_path"]


def test_auditor_forces_rewrite():
    """Código com paleta errada é rejeitado e o feedback chega ao gerador."""
    llm = FakeLLM(BAD_COLOR, GOOD)
    out = run_request("teste", llm, label="t_audit")
    assert out["status"] == "success" and out["iterations"] == 2
    assert "R1" in llm.prompts[1]                       # feedback incluído na reescrita
    assert out["history"][0]["approved"] is False


def test_runtime_error_forces_rewrite():
    llm = FakeLLM(BAD_RUNTIME, GOOD)
    out = run_request("teste", llm, label="t_run")
    assert out["status"] == "success" and out["iterations"] == 2
    assert "AttributeError" in llm.prompts[1]           # traceback incluído na reescrita


def test_gives_up_at_limit():
    out = run_request("teste", FakeLLM(BAD_COLOR), max_iterations=3, label="t_giveup")
    assert out["status"] == "failed" and out["iterations"] == 3
    assert "Desistiu" in out["error"]


def test_llm_failure_ends_cleanly(monkeypatch):
    class Boom:
        def invoke(self, _): raise ConnectionError("sem rede")
    # evita esperar o backoff real do tenacity durante o teste
    import tenacity
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)
    out = run_request("teste", Boom(), label="t_boom")
    assert out["status"] == "failed" and "Falha do LLM" in out["error"]


def test_routing_limit():
    st = {"audit": {"approved": False}, "iterations": 4, "max_iterations": 4}
    assert route_after_audit(st) == "give_up"
    st["iterations"] = 1
    assert route_after_audit(st) == "generate"


def test_graph_compiles_with_nodes():
    names = set(build_graph(FakeLLM(GOOD)).get_graph().nodes)
    assert {"generate", "audit", "execute", "give_up"} <= names
