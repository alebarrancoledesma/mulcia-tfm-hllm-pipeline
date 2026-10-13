"""Ejecuta los .rs de experiments/first_approach con GBFS y timeout largo.

Compila cada candidato una vez, lo ejecuta contra P/dev (el prompt vio
P-n16-k8 / P-n20-k2 / P-n23-k8: easy / medium / hard) y lo compara con
heurísticas built-in de PlanForge (blind, ff, lmcutnumeric).

Uso (desde WSL; PlanForge no compila en Windows):

    PYTHONPATH=src python3 scripts/try_first_approach.py
"""

from pathlib import Path

from hllm_pipeline import benchmark, data

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "first_approach"
CANDIDATES = sorted(RESULTS_ROOT.glob("*/out - *.rs"))
METRICS_PATH = RESULTS_ROOT / "gbfs_300s_metrics.csv"
COMPILE_LOG_DIR = RESULTS_ROOT / "compile_logs"

BUILTIN_HEURISTICS = ["blind", "ff", "lmcutnumeric"]
SEARCH = "gbfs"
MAX_TIME_S = 300

TEST_INSTANCE_IDS = data.load_split("P", "dev")
DOMAIN = data.domain_path()
INSTANCE_PATHS = data.instance_paths("P", TEST_INSTANCE_IDS)

MAX_MEMORY = "2G"

def _grounding_args(_instance_path):
    return []


def _failed_record(candidate, instance_path, error):
    return {
        "candidate": candidate,
        "instance": Path(instance_path).stem,
        "search": SEARCH,
        "solved": False,
        "runtime": None,
        "expanded": None,
        "evaluated": None,
        "generated": None,
        "solution_cost": None,
        "timeout": False,
        "error": error,
    }


def _save(record):
    print(record, flush=True)
    benchmark.append_metrics_row(record, METRICS_PATH)


if METRICS_PATH.exists():
    raise FileExistsError(f"{METRICS_PATH} ya existe; bórralo o renómbralo antes de relanzar")

COMPILE_LOG_DIR.mkdir(parents=True, exist_ok=True)

print(
    f"GBFS, timeout={MAX_TIME_S}s, {len(BUILTIN_HEURISTICS)} built-ins + "
    f"{len(CANDIDATES)} candidatos × {len(INSTANCE_PATHS)} instancias "
    f"({', '.join(TEST_INSTANCE_IDS)})",
    flush=True,
)
print(f"CSV incremental: {METRICS_PATH}", flush=True)

# 1) built-ins: un solo binario (planforge), corrido contra todas las instancias.
print("Compilando binario base de PlanForge...", flush=True)
compiled_planforge = benchmark.compile_planforge_binary()
if not compiled_planforge["compiled"]:
    print("No se pudo compilar el binario base de PlanForge:", flush=True)
    print(compiled_planforge["log"][-2000:], flush=True)
    log_path = COMPILE_LOG_DIR / "planforge.log"
    log_path.write_text(compiled_planforge["log"], encoding="utf-8")
    for heuristic in BUILTIN_HEURISTICS:
        for instance_path in INSTANCE_PATHS:
            _save(_failed_record(heuristic, instance_path, "compile_error: planforge base"))
else:
    print("PlanForge base compilado.", flush=True)
    for heuristic in BUILTIN_HEURISTICS:
        for instance_path in INSTANCE_PATHS:
            record = benchmark.run_candidate_on_instance(
                heuristic,
                DOMAIN,
                instance_path,
                binary_path=compiled_planforge["binary_path"],
                max_time_s=MAX_TIME_S,
                max_memory=MAX_MEMORY,
                search=SEARCH,
                extra_args=_grounding_args(instance_path),
            )
            record["candidate"] = heuristic
            _save(record)

# 2) cada candidato LLM: compilar una vez, correr contra todas las instancias.
for rs_path in CANDIDATES:
    label = f"{rs_path.parent.name} / {rs_path.stem}"
    log_name = f"{rs_path.parent.name}__{rs_path.stem}.log"
    print(f"Compilando {label}...", flush=True)
    compiled = benchmark.compile_candidate(rs_path.read_text(encoding="utf-8"))
    (COMPILE_LOG_DIR / log_name).write_text(compiled["log"], encoding="utf-8")

    if not compiled["compiled"]:
        print(f"{label}: NO COMPILA", flush=True)
        print(compiled["log"][-2000:], flush=True)
        error = f"compile_error: {compiled['log'][-1000:]}"
        for instance_path in INSTANCE_PATHS:
            _save(_failed_record(label, instance_path, error))
        continue

    print(f"{label}: compilado, ejecutando instancias...", flush=True)
    for instance_path in INSTANCE_PATHS:
        record = benchmark.run_candidate_on_instance(
            "llm_001",
            DOMAIN,
            instance_path,
            binary_path=compiled["binary_path"],
            max_time_s=MAX_TIME_S,
            max_memory=MAX_MEMORY,
            search=SEARCH,
            extra_args=_grounding_args(instance_path),
        )
        record["candidate"] = label
        _save(record)

print(f"\nGuardado en {METRICS_PATH}", flush=True)
