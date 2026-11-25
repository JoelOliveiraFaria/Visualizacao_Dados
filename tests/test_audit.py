"""Testes do auditor de cor (Fase 4): casos bons, maus e integração."""
import pytest

from src.agents.audit import audit_color, classify_cmap, load_profile, static_audit
from src.data_prep import ROOT

PROFILE = load_profile()


def rules(code):
    """Conjunto de (regra, severidade) devolvido para um snippet."""
    return {(f.rule, f.severity) for f in static_audit(code, PROFILE).findings}


# ---------------- casos que DEVEM ser reprovados ----------------
def test_r1_continuous_with_qualitative():
    code = "ax.scatter(d.gdpPercap, d.lifeExp, c=d.lifeExp, cmap='tab10')"
    r = static_audit(code, PROFILE)
    assert not r.approved and ("R1", "error") in rules(code)


def test_r1_seaborn_hue_continuous_palette_set1():
    assert ("R1", "error") in rules("sns.scatterplot(data=d, x='gdpPercap', y='lifeExp', hue='pop', palette='Set1')")


def test_r2_nominal_with_sequential():
    assert ("R2", "error") in rules("ax.scatter(d.x, d.y, c=d.continent, cmap='viridis')")


def test_r2_nominal_with_diverging():
    assert ("R2", "error") in rules("sns.scatterplot(data=d, hue='continent', palette='coolwarm')")


def test_r4_diverging_without_center():
    code = "ax.scatter(d.x, d.y, c=d.lifeExp, cmap='RdBu')"
    assert not static_audit(code, PROFILE).approved and ("R4", "error") in rules(code)


def test_r5_rainbow():
    assert ("R5", "error") in rules("ax.scatter(d.x, d.y, c=d.lifeExp, cmap='jet')")


def test_r5_reversed_rainbow_and_variable_indirection():
    assert ("R5", "error") in rules("cm = 'jet_r'\nax.scatter(d.x, d.y, c=d.pop, cmap=cm)")


def test_r5_get_cmap_form():
    assert ("R5", "error") in rules("ax.scatter(d.x, d.y, c=d.pop, cmap=plt.get_cmap('rainbow'))")


def test_r7_syntax_error():
    r = static_audit("def broken(:", PROFILE)
    assert not r.approved and r.findings[0].rule == "R7"


# ---------------- casos que DEVEM ser aprovados ----------------
@pytest.mark.parametrize("code", [
    "ax.scatter(d.x, d.y, c=d.lifeExp, cmap='viridis')",
    "ax.scatter(d.x, d.y, c=d.year, cmap=plt.cm.plasma)",
    "sns.scatterplot(data=d, x='gdpPercap', y='lifeExp', hue='continent', palette='colorblind')",
    "ax.scatter(d.x, d.y, c=d.continent.map(QUALITATIVE))",
    "ax.scatter(d.x, d.y, c=d.lifeExp, cmap='RdBu', norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))",
    "sns.heatmap(m, cmap='coolwarm', center=0)",
    "ax.scatter(d.x, d.y, color='red')",  # cor fixa, não mapeia variável
])
def test_valid_code_is_approved(code):
    r = static_audit(code, PROFILE)
    assert r.approved, r.feedback()


# ---------------- avisos ----------------
def test_r3_ordinal_with_qualitative_is_warning_only():
    code = "ax.scatter(d.x, d.y, c=d.year, cmap='tab10')"
    r = static_audit(code, PROFILE)
    assert r.approved and ("R3", "warning") in rules(code)


def test_r6_too_many_categories_warns():
    code = "sns.scatterplot(data=d, hue='country', palette='tab20')"
    assert ("R6", "warning") in rules(code)


# ---------------- classificação e feedback ----------------
@pytest.mark.parametrize("name,kind", [
    ("tab10", "qualitative"), ("Viridis", "sequential"), ("viridis_r", "sequential"),
    ("RdBu", "diverging"), ("jet", "rainbow"), ("turbo", "rainbow_warn"), ("xyz", "unknown"),
])
def test_classify_cmap(name, kind):
    assert classify_cmap(name) == kind


def test_feedback_contains_suggestion():
    fb = static_audit("ax.scatter(d.x, d.y, c=d.lifeExp, cmap='tab10')", PROFILE).feedback()
    assert "R1" in fb and "viridis" in fb


# ---------------- camada LLM (com modelo falso, sem rede) ----------------
class _FakeLLM:
    """Imita um modelo LangChain que devolve um veredicto fixo."""
    def __init__(self, verdict=None, boom=False):
        self.verdict, self.boom = verdict, boom

    def with_structured_output(self, schema):
        outer = self

        class _Chain:
            def invoke(self, prompt):
                if outer.boom:
                    raise RuntimeError("API em baixo")
                return schema(**outer.verdict)
        return _Chain()


def test_llm_error_finding_rejects():
    llm = _FakeLLM({"findings": [{"severity": "error", "message": "cores inacessíveis"}]})
    r = audit_color("ax.scatter(d.x, d.y, c=d.lifeExp, cmap='viridis')", PROFILE, llm)
    assert not r.approved


def test_llm_failure_does_not_block():
    r = audit_color("ax.scatter(d.x, d.y, c=d.lifeExp, cmap='viridis')", PROFILE, _FakeLLM(boom=True))
    assert r.approved and any(f.rule == "LLM" and f.severity == "info" for f in r.findings)


# ---------------- integração: o código real do projeto passa ----------------
def test_project_plots_pass_audit():
    code = (ROOT / "src" / "plots.py").read_text(encoding="utf-8")
    r = static_audit(code, PROFILE)
    assert r.approved, r.feedback()
