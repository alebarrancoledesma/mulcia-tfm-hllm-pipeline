"""Orquesta un experimento: config -> generar candidatos -> benchmark DEV -> rankear -> TEST.

Cada experimento vive en experiments/<exp_name>/ y nunca se sobrescribe
(crear un directorio o escribir un archivo que ya existe lanza excepción).
benchmark/rank/test son pasos explícitos, no automáticos, porque requieren
un checkout de PlanForge compilable (ver benchmark.py).
"""

from pathlib import Path

from hllm_pipeline import data
from hllm_pipeline.generate import generate_candidates
from hllm_pipeline.utils import write_file, write_json, read_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"


############################
# CONFIG
############################

def build_config(
    family,
    provider,
    model,
    num_candidates,
    reasoning_effort="medium",
    prompt_instance_ids=None,
    dev_instance_ids=None,
    test_instance_ids=None,
    name_prefix="h",
    seed=None,
    temperature=None,
    max_time_s=60,
):
    """Config del experimento; prompt/dev/test se toman del split si no se pasan."""
    prompt_instance_ids = prompt_instance_ids if prompt_instance_ids is not None else data.load_split(family, "prompt")
    dev_instance_ids = dev_instance_ids if dev_instance_ids is not None else data.load_split(family, "dev")
    test_instance_ids = test_instance_ids if test_instance_ids is not None else data.load_split(family, "test")

    return {
        "family": family,
        "provider": provider,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "num_candidates": num_candidates,
        "prompt_instance_ids": list(prompt_instance_ids),
        "dev_instance_ids": list(dev_instance_ids),
        "test_instance_ids": list(test_instance_ids),
        "name_prefix": name_prefix,
        "seed": seed,
        "temperature": temperature,
        "max_time_s": max_time_s,
    }


############################
# DIRECTORIO DE EXPERIMENTO
############################

def new_experiment_dir(exp_name):
    """Crea experiments/<exp_name>/ con sus subcarpetas; lanza excepción si ya existe."""
    exp_dir = EXPERIMENTS_ROOT / exp_name

    if exp_dir.exists():
        raise FileExistsError(f"El experimento ya existe: {exp_dir}")

    for sub in ("candidates", "prompts", "dev", "test"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=False)

    return exp_dir


def save_config(exp_dir, config):
    write_json(Path(exp_dir) / "config.json", config, overwrite=False)


def load_config(exp_dir):
    return read_json(Path(exp_dir) / "config.json")


############################
# PASO 1: GENERAR
############################

def run_generation(exp_dir, config):
    """Genera los candidatos del split PROMPT y guarda .rs/.json/.md de cada uno."""
    exp_dir = Path(exp_dir)
    domain_route = data.domain_path()
    instances_routes = data.instance_paths(config["family"], config["prompt_instance_ids"])

    candidates = generate_candidates(
        domain_route, instances_routes, config["provider"], config["model"], config["num_candidates"],
        llm_thinking=config["reasoning_effort"], name_prefix=config["name_prefix"],
    )

    names = []
    for candidate in candidates:
        name = candidate["name"]
        names.append(name)

        write_file(exp_dir / "candidates" / f"{name}.rs", candidate["code"], overwrite=False)
        write_file(exp_dir / "prompts" / f"{name}.md", candidate["prompt"], overwrite=False)
        write_json(
            exp_dir / "candidates" / f"{name}.json",
            {
                "name": name,
                "family": config["family"],
                "provider": config["provider"],
                "model": config["model"],
                "reasoning_effort": config["reasoning_effort"],
                "prompt_instance_ids": config["prompt_instance_ids"],
                **candidate["explainability"],
            },
            overwrite=False,
        )

    return names


############################
# PASO 2-3: BENCHMARK DEV + RANKING
############################

def _failed_records(candidate_name, instance_routes, reason):
    return [
        {
            "candidate": candidate_name, "instance": Path(p).stem, "search": None,
            "solved": False, "runtime": None, "expanded": None, "evaluated": None,
            "generated": None, "solution_cost": None, "timeout": False, "error": reason,
        }
        for p in instance_routes
    ]


def run_dev_benchmark(exp_dir, config, candidate_names, max_time_s=None):
    """Compila y ejecuta cada candidato en DEV; escribe dev/metrics.csv y devuelve los registros por candidato."""
    from hllm_pipeline import benchmark

    exp_dir = Path(exp_dir)
    max_time_s = max_time_s if max_time_s is not None else config.get("max_time_s", 60)
    domain_route = data.domain_path()
    dev_instance_routes = data.instance_paths(config["family"], config["dev_instance_ids"])

    dev_records_by_candidate = {}
    all_records = []

    for name in candidate_names:
        code = (exp_dir / "candidates" / f"{name}.rs").read_text(encoding="utf-8")
        compile_result = benchmark.compile_candidate(code)

        if not compile_result["compiled"]:
            records = _failed_records(name, dev_instance_routes, f"compile_error: {compile_result['log'][-1000:]}")
        else:
            records = benchmark.benchmark_candidate(
                name, domain_route, dev_instance_routes, binary_path=compile_result["binary_path"],
                max_time_s=max_time_s, search="gbfs",
            )

        dev_records_by_candidate[name] = records
        all_records.extend(records)

    benchmark.write_metrics_csv(all_records, exp_dir / "dev" / "metrics.csv")
    return dev_records_by_candidate


def run_ranking(exp_dir, csv_path=None):
    """Rankea el metrics.csv de DEV y escribe ranking.csv (incluye el seleccionado)."""
    from hllm_pipeline import rank as rank_module

    exp_dir = Path(exp_dir)
    csv_path = Path(csv_path) if csv_path else exp_dir / "metrics.csv"
    ranking = rank_module.rank_candidates(csv_path)
    out = csv_path.parent / "ranking.csv"
    ranking.to_csv(out, index=False)
    selected = ranking.iloc[0]["candidate"] if len(ranking) else None
    return {"ranking": ranking, "selected": selected, "path": out}


############################
# PASO 4: TEST
############################

def run_test_evaluation(exp_dir, config, selected_candidate, test_family=None, max_time_s=None):
    """Evalúa el candidato ya congelado en TEST (por defecto, la propia familia).

    `test_family` permite evaluar en la otra familia (estudio de transferencia).
    No selecciona ni rankea nada: `selected_candidate` debe venir ya de `ranking.json`.
    """
    from hllm_pipeline import benchmark

    exp_dir = Path(exp_dir)
    test_family = test_family or config["family"]
    domain_route = data.domain_path()
    test_instance_ids = data.load_split(test_family, "test") if test_family != config["family"] else config["test_instance_ids"]
    test_instance_routes = data.instance_paths(test_family, test_instance_ids)

    code = (exp_dir / "candidates" / f"{selected_candidate}.rs").read_text(encoding="utf-8")
    compile_result = benchmark.compile_candidate(code)
    max_time_s = max_time_s if max_time_s is not None else config.get("max_time_s", 60)

    if not compile_result["compiled"]:
        records = _failed_records(selected_candidate, test_instance_routes, f"compile_error: {compile_result['log'][-1000:]}")
    else:
        records = benchmark.benchmark_candidate(
            selected_candidate, domain_route, test_instance_routes, binary_path=compile_result["binary_path"],
            max_time_s=max_time_s, search="gbfs",
        )

    benchmark.write_metrics_csv(records, exp_dir / "test" / f"metrics_{test_family}.csv")
    return records


############################
# ORQUESTACIÓN COMPLETA
############################

def run_experiment(exp_name, config, do_dev_benchmark=False, do_test=False):
    """Genera y, si se pide, encadena benchmark DEV + ranking + TEST."""
    exp_dir = new_experiment_dir(exp_name)
    save_config(exp_dir, config)
    candidate_names = run_generation(exp_dir, config)

    if not do_dev_benchmark:
        return {"exp_dir": exp_dir, "candidates": candidate_names}

    run_dev_benchmark(exp_dir, config, candidate_names)
    ranking = run_ranking(exp_dir)

    if not do_test or ranking["selected"] is None:
        return {"exp_dir": exp_dir, "candidates": candidate_names, "ranking": ranking}

    test_records = run_test_evaluation(exp_dir, config, ranking["selected"])
    return {"exp_dir": exp_dir, "candidates": candidate_names, "ranking": ranking, "test_records": test_records}
