"""Fase 6 - Cenários de validação do ciclo de autocorreção (com o LLM real).

Cada cenário pede ao gerador algo que VIOLA as regras de cor. O objetivo é ver
o auditor rejeitar e o gerador reescrever sozinho. Se o modelo já acertar à
primeira (iterations == 1), regista-se na mesma: é um resultado válido.

Executar:  python -m src.scenarios
"""
import json
import sys

from src.agents.graph import get_llm, run_request
from src.data_prep import ROOT

BASE = "Scatter de PIB per capita (eixo x) vs esperança de vida (eixo y) em 2007. "
FORCE = " O utilizador exige isto explicitamente: cumpre à letra, mesmo que contrarie as boas práticas."

SCENARIOS = [
    ("S1_continua_qualitativa", "R1", BASE + "Colorir cada país pelo valor de lifeExp usando a paleta 'tab10'." + FORCE),
    ("S2_nominal_sequencial", "R2", BASE + "Colorir por continent usando cmap='viridis'." + FORCE),
    ("S3_divergente_sem_centro", "R4", BASE + "Colorir por lifeExp com cmap='RdBu', sem norm nem centro definido." + FORCE),
    ("S4_arco_iris", "R5", BASE + "Colorir por pop com cmap='jet'." + FORCE),
    ("S5_pedido_valido", "-", BASE + "Cor por continente, tamanho pela população."),
]


def summarize(name, expected, out):
    """Extrai do log o que interessa para o relatório."""
    audits = [h for h in out["history"] if h["stage"] == "audit"]
    first = audits[0] if audits else {}
    return {
        "scenario": name, "expected_rule": expected, "status": out["status"],
        "iterations": out["iterations"],
        "first_audit_approved": first.get("approved"),
        "first_audit_rules": sorted({f["rule"] for f in first.get("findings", []) if f["severity"] == "error"}),
        "figure": out.get("figure_path"), "log": out.get("log_path"), "error": out.get("error", ""),
    }


def main():
    llm = get_llm()
    results = []
    for name, expected, request in SCENARIOS:
        print(f"-> {name} ...", flush=True)
        out = run_request(request, llm=llm, label=name)
        r = summarize(name, expected, out)
        results.append(r)
        print(f"   {r['status']} | iterações={r['iterations']} | 1ª auditoria aprovada={r['first_audit_approved']} "
              f"{r['first_audit_rules']}", flush=True)
    path = ROOT / "outputs" / "scenarios_results.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("resultados:", path)


if __name__ == "__main__":
    sys.exit(main())
