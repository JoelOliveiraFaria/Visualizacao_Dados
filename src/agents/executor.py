"""Fase 3 - Executor: corre o código gerado pelo LLM num subprocesso isolado.

Porquê subprocesso e não exec():
  - um erro/ciclo infinito no código gerado não derruba o grafo (há timeout);
  - o stderr (traceback) é devolvido ao agente gerador para ele se corrigir;
  - o estado do matplotlib não "vaza" entre iterações.

Nota: isto NÃO é uma sandbox de segurança (o código tem acesso ao disco). É
adequado para um trabalho académico com código gerado a partir de prompts
controlados; num produto real usar contentor/VM.
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.data_prep import ROOT

GENERATED_DIR = ROOT / "outputs" / "generated"  # scripts gerados (rastreabilidade)
FIGURES_DIR = ROOT / "outputs" / "figures"
MAX_STDERR_CHARS = 1500  # o traceback é cortado para caber no prompt


@dataclass
class ExecResult:
    ok: bool                 # True se correu sem erro E gravou a figura
    figure_path: str | None  # caminho da figura gerada
    error: str               # traceback/mensagem (vazio se ok)
    script_path: str         # script executado (para o relatório)


def run_code(code: str, name: str = "generated", timeout: int = 90) -> ExecResult:
    """Executa `code` e devolve o resultado.

    O código recebe, pré-definidos, o backend 'Agg' e a variável OUTPUT_PATH
    (onde DEVE gravar a figura com plt.savefig(OUTPUT_PATH)).
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    out_path = FIGURES_DIR / f"{name}_{run_id}.png"
    script = GENERATED_DIR / f"{name}_{run_id}.py"

    # Prelúdio: garante backend sem janela e diz ao código onde gravar.
    prelude = f"import matplotlib\nmatplotlib.use('Agg')\nOUTPUT_PATH = {str(out_path)!r}\n"
    script.write_text(prelude + code, encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return ExecResult(False, None, f"Timeout: o código excedeu {timeout}s (ciclo infinito?).", str(script))

    if proc.returncode != 0:
        return ExecResult(False, None, proc.stderr[-MAX_STDERR_CHARS:], str(script))
    if not out_path.exists():
        return ExecResult(False, None, "O código correu mas não gravou a figura. "
                          "Usar plt.savefig(OUTPUT_PATH, bbox_inches='tight').", str(script))
    return ExecResult(True, str(out_path), "", str(script))
