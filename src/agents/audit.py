"""Fase 4 - Nó de auditoria de cor.

Inspeciona o CÓDIGO gerado (sem o executar) e valida se a escala de cor usada
é adequada à natureza da variável mapeada em cor (contínua vs. categórica).

Duas camadas:
  1. Estática (determinística, via `ast`): regras fixas, rápidas e testáveis.
  2. LLM-as-judge (opcional): cobre casos que as regras não apanham.

Como funciona a camada estática, para cada chamada de função no código:
  - procura um argumento "canal de cor" (c=, hue=, color=) que referencie uma
    coluna do dataset (ex.: d.continent, d["year"], hue="continent");
  - procura o argumento da paleta (cmap=, palette=) e classifica-o em
    qualitativa / sequencial / divergente / arco-íris;
  - cruza as duas coisas com a natureza da variável (variable_profile.json).

Regras (severidade "error" reprova o código, "warning" apenas avisa):
  R1  variável contínua  + paleta qualitativa            -> error
  R2  variável nominal   + paleta sequencial/divergente  -> error
  R3  variável ordinal   + paleta qualitativa            -> warning
  R4  paleta divergente sem centro (norm/center/vcenter) -> error
  R5  paleta arco-íris (jet, rainbow, ...)               -> error
  R6  paleta qualitativa com > 8 categorias              -> warning
  R7  código com erro de sintaxe                          -> error
"""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field

from src.data_prep import PROFILE
from src import palettes

# ---------------------------------------------------------------------------
# Dicionários de conhecimento: nome de colormap -> tipo de paleta
# ---------------------------------------------------------------------------
QUALITATIVE_CMAPS = {
    "tab10", "tab20", "tab20b", "tab20c", "set1", "set2", "set3", "paired",
    "accent", "dark2", "pastel1", "pastel2",
    # paletas qualitativas do seaborn
    "deep", "muted", "bright", "pastel", "dark", "colorblind",
}
SEQUENTIAL_CMAPS = {
    "viridis", "plasma", "inferno", "magma", "cividis", "blues", "greens",
    "reds", "greys", "oranges", "purples", "ylorrd", "ylgnbu", "ylorbr",
    "ylgn", "bugn", "bupu", "gnbu", "orrd", "pubu", "pubugn", "purd", "rdpu",
    "binary", "gray", "rocket", "mako", "flare", "crest",
}
DIVERGING_CMAPS = {
    "rdbu", "rdylbu", "rdylgn", "rdgy", "coolwarm", "bwr", "seismic",
    "piyg", "prgn", "brbg", "puor", "spectral", "vlag", "icefire",
}
RAINBOW_CMAPS = {"jet", "rainbow", "hsv", "nipy_spectral", "gist_rainbow", "gist_ncar"}
RAINBOW_WARN = {"turbo"}  # rainbow melhorado: só aviso

# Funções do seaborn/matplotlib que criam paletas; mapeadas para o tipo.
PALETTE_BUILDERS = {
    "light_palette": "sequential", "dark_palette": "sequential",
    "cubehelix_palette": "sequential", "diverging_palette": "diverging",
}

# Argumentos que ligam uma variável à cor / argumentos que definem a paleta.
COLOR_CHANNEL_KWARGS = {"c", "hue", "color", "facecolors"}
PALETTE_KWARGS = {"cmap", "palette", "colormap"}
CENTER_KWARGS = {"norm", "center", "vcenter"}  # fixam o centro da escala divergente

MAX_QUALITATIVE_LEVELS = 8  # Wilke cap. 4: acima disto as cores confundem-se

# Constantes de src/palettes.py que o código gerado costuma importar.
KNOWN_NAMES = {
    "SEQUENTIAL": palettes.SEQUENTIAL,
    "DIVERGING": palettes.DIVERGING,
    "QUALITATIVE": "qualitative",  # é um dict categoria->cor
}

# Paleta recomendada por natureza (usada nas sugestões ao agente gerador).
RECOMMENDED = {
    "categorical_nominal": "paleta qualitativa (QUALITATIVE / Okabe-Ito)",
    "ordinal_temporal": "paleta sequencial (cmap='viridis')",
    "quantitative_continuous": "paleta sequencial (cmap='viridis'); divergente só com centro (TwoSlopeNorm)",
}


# ---------------------------------------------------------------------------
# Estruturas de resultado
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    """Uma violação ou aviso encontrado no código."""
    rule: str            # ex.: "R1"
    severity: str        # "error" | "warning" | "info"
    line: int            # linha do código (0 se não aplicável)
    message: str         # o que está mal
    suggestion: str = "" # como corrigir (vai para o prompt do gerador)


@dataclass
class AuditResult:
    """Veredicto da auditoria."""
    approved: bool
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def to_dict(self) -> dict:
        return {"approved": self.approved, "findings": [asdict(f) for f in self.findings]}

    def feedback(self) -> str:
        """Texto para devolver ao agente gerador, a pedir reescrita."""
        if not self.findings:
            return "Auditoria de cor aprovada."
        lines = []
        for f in self.findings:
            if f.severity == "info":
                continue
            lines.append(f"[{f.rule}/{f.severity}] linha {f.line}: {f.message}"
                         + (f" -> {f.suggestion}" if f.suggestion else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classificação de paletas
# ---------------------------------------------------------------------------
def classify_cmap(name: str) -> str:
    """Devolve 'qualitative' | 'sequential' | 'diverging' | 'rainbow' | 'rainbow_warn' | 'unknown'."""
    n = name.lower()
    if n.endswith("_r"):  # versão invertida do mesmo colormap
        n = n[:-2]
    if n in RAINBOW_CMAPS:
        return "rainbow"
    if n in RAINBOW_WARN:
        return "rainbow_warn"
    if n in QUALITATIVE_CMAPS:
        return "qualitative"
    if n in DIVERGING_CMAPS:
        return "diverging"
    if n in SEQUENTIAL_CMAPS:
        return "sequential"
    return "unknown"


class _Auditor(ast.NodeVisitor):
    """Percorre a árvore sintática e aplica as regras R1-R6."""

    def __init__(self, profile: dict):
        self.profile = profile
        self.findings: list[Finding] = []
        # Ambiente simples: nomes -> valor constante (ex.: cm = "jet").
        self.env: dict[str, str] = dict(KNOWN_NAMES)

    # -- recolha de atribuições simples (cm = "viridis") ---------------------
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.env[t.id] = node.value.value
        self.generic_visit(node)

    # -- análise de cada chamada --------------------------------------------
    def visit_Call(self, node: ast.Call):
        kws = {k.arg: k.value for k in node.keywords if k.arg}
        self._check_call(node, kws)
        self.generic_visit(node)

    def _check_call(self, node: ast.Call, kws: dict[str, ast.expr]):
        # 1) tipo de paleta usado neste call (cmap=/palette=)
        palette_kind, palette_name = None, None
        for key in PALETTE_KWARGS:
            if key in kws:
                palette_kind, palette_name = self._resolve_palette(kws[key])
                break

        # 2) variável mapeada em cor (c=/hue=/color=) e, se houver, mapeamento explícito
        column, explicit_qualitative = None, False
        for key in COLOR_CHANNEL_KWARGS:
            if key in kws:
                column = column or self._column_in(kws[key])
                explicit_qualitative = explicit_qualitative or self._has_dict_mapping(kws[key])
        if palette_kind is None and explicit_qualitative:
            palette_kind, palette_name = "qualitative", "mapeamento explícito (dict)"

        if palette_kind is None:
            return  # nada a auditar neste call (o default do matplotlib é sequencial)
        line = node.lineno

        # R5 - arco-íris
        if palette_kind == "rainbow":
            self._add("R5", "error", line, f"Paleta arco-íris '{palette_name}' não é perceptualmente uniforme.",
                      "Usar 'viridis' (sequencial) ou uma paleta divergente com centro.")
        elif palette_kind == "rainbow_warn":
            self._add("R5", "warning", line, f"Paleta '{palette_name}' tem aspeto arco-íris.",
                      "Preferir 'viridis'.")

        # R4 - divergente sem centro
        if palette_kind == "diverging" and not (CENTER_KWARGS & kws.keys()):
            self._add("R4", "error", line,
                      f"Paleta divergente '{palette_name}' sem centro definido.",
                      "Passar norm=TwoSlopeNorm(vcenter=...) (ou center= no seaborn) com um valor neutro com significado.")

        if column is None:
            return  # sem variável identificável: não dá para cruzar com a natureza
        info = self.profile[column]
        nature = info["nature"]

        # R1 - contínua + qualitativa
        if nature == "quantitative_continuous" and palette_kind == "qualitative":
            self._add("R1", "error", line,
                      f"'{column}' é contínua mas está mapeada com paleta qualitativa ({palette_name}).",
                      f"Usar {RECOMMENDED[nature]}.")
        # R2 - nominal + sequencial/divergente
        if nature == "categorical_nominal" and palette_kind in ("sequential", "diverging"):
            self._add("R2", "error", line,
                      f"'{column}' é categórica nominal mas está mapeada com paleta {palette_kind} ({palette_name}), "
                      "que sugere uma ordem inexistente.",
                      f"Usar {RECOMMENDED[nature]}.")
        # R3 - ordinal + qualitativa
        if nature == "ordinal_temporal" and palette_kind == "qualitative":
            self._add("R3", "warning", line,
                      f"'{column}' é ordenada mas usa paleta qualitativa ({palette_name}): perde-se a ordem.",
                      f"Preferir {RECOMMENDED[nature]}.")
        # R6 - demasiadas categorias
        if palette_kind == "qualitative" and info["cardinality"] > MAX_QUALITATIVE_LEVELS \
                and nature == "categorical_nominal":
            self._add("R6", "warning", line,
                      f"'{column}' tem {info['cardinality']} categorias; mais de {MAX_QUALITATIVE_LEVELS} cores distintas "
                      "não se distinguem bem.",
                      "Agrupar categorias, filtrar as principais ou usar outra codificação (facetas, dot plot).")

    # -- utilitários --------------------------------------------------------
    def _add(self, rule, severity, line, message, suggestion=""):
        self.findings.append(Finding(rule, severity, line, message, suggestion))

    def _column_in(self, expr: ast.expr) -> str | None:
        """Primeira coluna do dataset referenciada na expressão (d.col, d["col"], "col")."""
        for n in ast.walk(expr):
            if isinstance(n, ast.Attribute) and n.attr in self.profile:
                return n.attr
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                    and n.slice.value in self.profile:
                return n.slice.value
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in self.profile:
                return n.value
        return None

    def _has_dict_mapping(self, expr: ast.expr) -> bool:
        """Deteta .map(QUALITATIVE) ou .map({...}): cor atribuída por dicionário (categórica)."""
        for n in ast.walk(expr):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "map" and n.args:
                a = n.args[0]
                if isinstance(a, ast.Dict) or (isinstance(a, ast.Name) and self.env.get(a.id) == "qualitative"):
                    return True
        return False

    def _resolve_palette(self, expr: ast.expr) -> tuple[str, str]:
        """Devolve (tipo, nome) da paleta indicada numa expressão."""
        if isinstance(expr, (ast.Dict, ast.List, ast.Tuple)):  # lista/dict de cores
            return "qualitative", "lista/dict de cores"
        name = self._name_of(expr)
        if name is None:
            return "unknown", "?"
        if name in PALETTE_BUILDERS:  # sns.light_palette(...) etc.
            return PALETTE_BUILDERS[name], name
        if name == "qualitative":
            return "qualitative", "QUALITATIVE"
        return classify_cmap(name), name

    def _name_of(self, expr: ast.expr) -> str | None:
        """Extrai o nome do colormap de: "x", NOME, plt.cm.x, get_cmap("x"), color_palette("x")."""
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.Name):
            return self.env.get(expr.id)
        if isinstance(expr, ast.Attribute):  # plt.cm.viridis
            return expr.attr
        if isinstance(expr, ast.Call):
            fname = expr.func.attr if isinstance(expr.func, ast.Attribute) else getattr(expr.func, "id", "")
            if fname in PALETTE_BUILDERS:
                return fname
            if expr.args:
                return self._name_of(expr.args[0])  # get_cmap("viridis"), color_palette("Set1")
        return None


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def load_profile() -> dict:
    """Carrega o dicionário de dados gerado por data_prep.py."""
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def static_audit(code: str, profile: dict | None = None) -> AuditResult:
    """Camada 1: auditoria estática determinística via AST."""
    profile = profile or load_profile()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:  # R7
        return AuditResult(False, [Finding("R7", "error", e.lineno or 0, f"Erro de sintaxe: {e.msg}",
                                           "Corrigir a sintaxe do código.")])
    auditor = _Auditor(profile)
    auditor.visit(tree)
    approved = not any(f.severity == "error" for f in auditor.findings)
    return AuditResult(approved, auditor.findings)


def audit_color(code: str, profile: dict | None = None, llm=None) -> AuditResult:
    """Auditoria completa: camada estática + (opcional) juiz LLM.

    `llm` é qualquer modelo LangChain com `.with_structured_output`. Se for None,
    só corre a camada estática (rápida, sem custo, determinística).
    """
    profile = profile or load_profile()
    result = static_audit(code, profile)
    if llm is not None:
        result = _merge(result, llm_judge(code, profile, llm))
    return result


# ---------------------------------------------------------------------------
# Camada 2: LLM-as-judge
# ---------------------------------------------------------------------------
try:  # pydantic vem com o langchain; falha suave se não existir
    from pydantic import BaseModel, Field

    class JudgeFinding(BaseModel):
        severity: str = Field(description="'error' ou 'warning'")
        message: str
        suggestion: str = ""

    class JudgeVerdict(BaseModel):
        """Resposta estruturada do juiz."""
        findings: list[JudgeFinding] = Field(default_factory=list)
except ImportError:  # pragma: no cover
    JudgeVerdict = None

JUDGE_PROMPT = """És um auditor de visualização de dados (Wilke, "Fundamentals of Data Visualization", cap. 4).
Avalia SÓ o uso de COR no código abaixo. Verifica se o tipo de escala de cor é adequado à natureza
de cada variável mapeada em cor:
- categórica nominal -> paleta qualitativa;
- contínua/ordenada  -> paleta sequencial perceptualmente uniforme;
- divergente apenas se existir um centro com significado, e esse centro tem de estar definido.
Aponta também cores não acessíveis a daltónicos ou arco-íris. Não repitas problemas triviais de estilo.
Devolve apenas problemas reais (lista vazia se estiver tudo bem).

Dicionário de dados (natureza de cada variável):
{profile}

Código:
```python
{code}
```"""


def llm_judge(code: str, profile: dict, llm) -> list[Finding]:
    """Pede ao LLM um veredicto estruturado. Falhas do LLM não bloqueiam o fluxo."""
    slim = {k: {"nature": v["nature"], "cardinality": v["cardinality"]} for k, v in profile.items()}
    prompt = JUDGE_PROMPT.format(profile=json.dumps(slim, ensure_ascii=False), code=code)
    try:
        verdict = llm.with_structured_output(JudgeVerdict).invoke(prompt)
    except Exception as e:  # rede, quota, formato... o grafo decide o que fazer
        return [Finding("LLM", "info", 0, f"Juiz LLM indisponível: {type(e).__name__}")]
    out = []
    for f in verdict.findings:
        sev = f.severity if f.severity in ("error", "warning") else "warning"
        out.append(Finding("LLM", sev, 0, f.message, f.suggestion))
    return out


def _merge(result: AuditResult, extra: list[Finding]) -> AuditResult:
    findings = result.findings + extra
    return AuditResult(not any(f.severity == "error" for f in findings), findings)
