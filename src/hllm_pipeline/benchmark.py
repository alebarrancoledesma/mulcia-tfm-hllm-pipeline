"""Compila y ejecuta una heurística en PlanForge, y extrae métricas de su log.

Usa el runner `experiments/llm-cvrp-runner` del repo PlanForge: el candidato
se escribe como `main.rs` de ese crate y se ejecuta con
`--search 'gbfs(nombre())'` (o `astar(...)`) domain.pddl problem.pddl. Las métricas salen de
parsear las líneas de log de `planforge/src/output.rs` (Expanded/Evaluated/
Search time/Plan cost/Solution found); no hay API JSON todavía, así que lo
que no se puede leer de ahí se deja en `None` en vez de inventarse.
"""

import csv
import os
import re
import subprocess
import time
from pathlib import Path

from hllm_pipeline.utils import write_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# El checkout de PlanForge es, por defecto, un repo hermano; override con PLANFORGE_ROOT.
PLANFORGE_ROOT = Path(os.environ.get("PLANFORGE_ROOT", PROJECT_ROOT.parent / "mulcia-tfm-PlanForge"))
RUNNER_CRATE_NAME = "planforge-llm-cvrp"
RUNNER_CRATE_DIR = PLANFORGE_ROOT / "experiments" / "llm-cvrp-runner"
RUNNER_MAIN_RS = RUNNER_CRATE_DIR / "src" / "main.rs"

DEFAULT_METRICS_FIELDS = [
    "candidate", "instance", "search", "solved", "runtime", "expanded", "evaluated",
    "generated", "solution_cost", "timeout", "error",
]

_PATTERNS = {
    "expanded": re.compile(r"Expanded (\d+) state"),
    "evaluated": re.compile(r"Evaluated (\d+) state"),
    "generated": re.compile(r"Generated (\d+) state"),
    "search_time": re.compile(r"Search time: ([\d.]+)s"),
    "plan_cost": re.compile(r"Plan cost: ([\d.]+)"),
}


def _binary_path():
    exe = "planforge-llm-cvrp.exe" if os.name == "nt" else "planforge-llm-cvrp"
    return PLANFORGE_ROOT / "target" / "release" / exe


def planforge_binary_path():
    exe = "planforge.exe" if os.name == "nt" else "planforge"
    return PLANFORGE_ROOT / "target" / "release" / exe


def compile_planforge_binary(timeout=600):
    """Compila el binario base de PlanForge (heurísticas built-in: blind, ff, lmcutnumeric, ...)."""
    try:
        result = subprocess.run(
            ["cargo", "build", "--release", "-p", "planforge"],
            cwd=PLANFORGE_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return {"compiled": False, "binary_path": None, "log": f"cargo build agotó el tiempo: {e}"}

    log = (result.stdout or "") + (result.stderr or "")
    compiled = result.returncode == 0 and planforge_binary_path().exists()

    return {"compiled": compiled, "binary_path": planforge_binary_path() if compiled else None, "log": log}


def compile_candidate(candidate_code, timeout=600):
    """Escribe el candidato como main.rs del runner y lo compila con cargo.

    No es paralelizable entre candidatos (crate compartido): compilar y
    ejecutar debe hacerse de uno en uno.
    """
    if not RUNNER_CRATE_DIR.exists():
        return {
            "compiled": False,
            "binary_path": None,
            "log": f"No existe el crate runner en {RUNNER_CRATE_DIR} (ver PLANFORGE_ROOT)",
        }

    write_file(RUNNER_MAIN_RS, candidate_code, overwrite=True)

    try:
        result = subprocess.run(
            ["cargo", "build", "--release", "-p", RUNNER_CRATE_NAME],
            cwd=PLANFORGE_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return {"compiled": False, "binary_path": None, "log": f"cargo build agotó el tiempo: {e}"}

    log = (result.stdout or "") + (result.stderr or "")
    compiled = result.returncode == 0 and _binary_path().exists()

    return {"compiled": compiled, "binary_path": _binary_path() if compiled else None, "log": log}


def _parse_run_output(text):
    parsed = {"expanded": None, "evaluated": None, "generated": None, "solution_cost": None}

    for key in ("expanded", "evaluated", "generated"):
        m = _PATTERNS[key].search(text)
        if m:
            parsed[key] = int(m.group(1))

    m = _PATTERNS["plan_cost"].search(text)
    if m:
        parsed["solution_cost"] = float(m.group(1))

    if "Solution found!" in text:
        parsed["solved"] = True
    elif "No solution found" in text or "Search timed out" in text:
        parsed["solved"] = False
    else:
        parsed["solved"] = None  # salida inesperada, no adivinamos

    return parsed


def run_candidate_on_instance(
    candidate_name,
    domain_path,
    problem_path,
    binary_path=None,
    max_time_s=300,
    max_memory=None,
    search="gbfs",
    extra_args=None,
):
    """Ejecuta un candidato compilado sobre una instancia y devuelve sus métricas.

    `max_time_s` se pasa como `--max-time` a PlanForge (el límite lo aplica
    el propio planner, no solo el timeout del subprocess).
    `search` es el algoritmo (`gbfs` o `astar`); la heurística va dentro:
    `gbfs(llm_001())`, `gbfs(ff())`, etc.
    `extra_args` se inserta antes de domain/problem (p. ej. límites de grounding).
    """
    binary_path = binary_path or _binary_path()

    record = {
        "candidate": candidate_name, "instance": Path(problem_path).stem, "search": search,
        "solved": False, "runtime": None, "expanded": None, "evaluated": None,
        "generated": None, "solution_cost": None, "timeout": False, "error": None,
    }

    if not Path(binary_path).exists():
        record["error"] = f"binary not found: {binary_path}"
        return record

    args = [str(binary_path), "--search", f"{search}({candidate_name}())"]
    if max_time_s is not None:
        args += ["--max-time", f"{max_time_s}s"]
    if max_memory is not None:
        args += ["--max-memory", str(max_memory)]
    if extra_args:
        args += list(extra_args)
    args += [str(domain_path), str(problem_path)]

    start = time.monotonic()
    try:
        # Margen extra para grounding (instancias medianas/grandes) + cierre.
        subprocess_timeout = max_time_s + 600 if max_time_s else None
        result = subprocess.run(args, capture_output=True, text=True, timeout=subprocess_timeout)
    except subprocess.TimeoutExpired:
        record["timeout"] = True
        record["runtime"] = time.monotonic() - start
        record["error"] = "timeout de subprocess (el planner no se autoterminó)"
        return record

    wall_time = time.monotonic() - start
    output_text = (result.stdout or "") + (result.stderr or "")

    if result.returncode == 7:
        record["timeout"] = True
        record["runtime"] = wall_time
        return record
    if result.returncode == 6:
        record["error"] = "out of memory"
        record["runtime"] = wall_time
        return record
    if result.returncode != 0:
        record["error"] = f"exit code {result.returncode}: {output_text[-2000:]}"
        record["runtime"] = wall_time
        return record

    record.update(_parse_run_output(output_text))
    m = _PATTERNS["search_time"].search(output_text)
    record["runtime"] = float(m.group(1)) if m else wall_time
    if record["solved"] is None:
        record["error"] = "no se pudo parsear la salida de PlanForge"

    return record


def benchmark_candidate(
    candidate_name,
    domain_path,
    instance_paths,
    binary_path=None,
    max_time_s=300,
    max_memory=None,
    search="gbfs",
    extra_args=None,
):
    return [
        run_candidate_on_instance(
            candidate_name,
            domain_path,
            p,
            binary_path=binary_path,
            max_time_s=max_time_s,
            max_memory=max_memory,
            search=search,
            extra_args=extra_args,
        )
        for p in instance_paths
    ]


def write_metrics_csv(records, path, fields=None):
    """Escribe métricas a CSV; lanza excepción si el archivo ya existe."""
    fields = fields or DEFAULT_METRICS_FIELDS
    path = Path(path)

    if path.exists():
        raise FileExistsError(f"{path} ya existe, no se sobrescribe")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in fields})


def append_metrics_row(record, path, fields=None):
    """Añade una fila a un CSV de métricas; crea el archivo con cabecera si no existe."""
    fields = fields or DEFAULT_METRICS_FIELDS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k) for k in fields})
