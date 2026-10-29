"""Genera N candidatos en paralelo con OpenAI gpt-5.6-sol + main_prompt.

Cada candidato va a outputs/numeric-planning/<EXP>/01/, 02/, ... con:
    out.rs       — código Rust
    out.json     — what / why / ...
    prompt.md    — prompt enviado al LLM

Uso:
    uv run python -u scripts/gen_candidates.py experiment001 --split P/prompt
    uv run python -u scripts/gen_candidates.py experiment002 --split E/prompt
    uv run python -u scripts/gen_candidates.py experiment002 --split E/prompt --n 10
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

from hllm_pipeline.data import domain_path, instance_path, split_path
from hllm_pipeline.llm import openai_generate, prompt_gen
from hllm_pipeline.utils import read_text_file, write_file, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROVIDER    = "openai"
MODEL       = "gpt-5.6-sol"
THINKING    = "medium"
PROMPT_FILE = PROJECT_ROOT / "prompts" / "main_prompt.md"


def resolve_split(spec: str) -> dict:
    """Acepta ruta a un JSON de split o 'P/prompt' / 'E/dev'."""
    path = Path(spec)
    if not path.is_file():
        alt = PROJECT_ROOT / spec
        if alt.is_file():
            path = alt
        elif "/" in spec or "\\" in spec:
            family, split = spec.replace("\\", "/").split("/", 1)
            path = split_path(family, split)
        else:
            path = split_path(spec, "prompt")
    if not path.is_file():
        raise FileNotFoundError(f"split no encontrado: {spec}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "family" not in data or "instances" not in data:
        raise ValueError(f"{path} debe tener 'family' e 'instances'")
    data["_path"] = str(path)
    return data


def build_prompt(family: str, instance_ids: list[str]) -> str:
    domain_pddl = read_text_file(domain_path())
    instances_pddl = "\n\n".join(
        f"## Representative instance {n+1}\n```pddl\n{read_text_file(instance_path(family, iid))}```"
        for n, iid in enumerate(instance_ids)
    )
    template = read_text_file(PROJECT_ROOT / "templates" / "heuristic_template.rs")
    return prompt_gen(PROMPT_FILE, {
        "domain_pddl":            domain_pddl,
        "representative_instances": instances_pddl,
        "heuristic_template":     template,
    })


def call_llm(idx: int, prompt: str) -> tuple[int, str]:
    """Llama a OpenAI y devuelve (idx, raw_response)."""
    print(f"[{idx:02d}] enviando...", flush=True)
    raw = openai_generate(prompt, model=MODEL, reasoning_effort=THINKING)
    print(f"[{idx:02d}] respuesta recibida ({len(raw)} chars)", flush=True)
    return idx, raw


def save_candidate(out_dir: Path, prompt: str, raw: str) -> bool:
    """Parsea la respuesta y guarda los 3 ficheros. Devuelve True si ok."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ERROR JSON: {e}", flush=True)
        (out_dir / "out.raw.txt").write_text(raw, encoding="utf-8")
        return False

    code = payload.pop("code", "")
    if not code.strip():
        print(f"  ERROR: campo 'code' vacío", flush=True)
        (out_dir / "out.raw.txt").write_text(raw, encoding="utf-8")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    write_file(out_dir / "prompt.md",  prompt, overwrite=True)
    write_file(out_dir / "out.rs",     code,   overwrite=True)
    write_json(out_dir / "out.json",   payload, overwrite=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Nombre de la ejecución")
    parser.add_argument(
        "--split",
        default="P/prompt",
        help="JSON de split o atajo P/prompt, E/prompt (default: P/prompt)",
    )
    parser.add_argument("--n",   type=int, default=10, help="Número de candidatos (default 10)")
    parser.add_argument("--rpm", type=int, default=20, help="Requests per minute para el sleep (default 20)")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "numeric-planning",
        help="Directorio que contendrá la carpeta de la ejecución",
    )
    args = parser.parse_args()

    split = resolve_split(args.split)
    family = split["family"]
    instance_ids = split["instances"]

    out_root = args.output.resolve() / args.experiment
    n        = args.n
    rpm      = args.rpm
    sleep_s  = 60.0 / rpm

    print(f"Experimento : {args.experiment}", flush=True)
    print(f"Split       : {split['_path']}", flush=True)
    print(f"Familia     : {family}  {instance_ids}", flush=True)
    print(f"Candidatos  : {n}", flush=True)
    print(f"Modelo      : {MODEL} (reasoning={THINKING})", flush=True)
    print(f"Prompt      : {PROMPT_FILE.name}", flush=True)
    print(f"Salida      : {out_root}", flush=True)
    print(f"RPM sleep   : {sleep_s:.1f}s entre envíos", flush=True)
    print(flush=True)

    print("Construyendo prompt...", flush=True)
    prompt = build_prompt(family, instance_ids)
    print(f"Prompt listo ({len(prompt)} chars)\n", flush=True)

    # Lanzar N llamadas en paralelo con sleep entre envíos
    futures: dict[concurrent.futures.Future, int] = {}
    results: dict[int, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        for idx in range(1, n + 1):
            # Saltar si ya existe
            out_dir = out_root / f"{idx:02d}"
            if (out_dir / "out.rs").exists():
                print(f"[{idx:02d}] SKIP (ya existe)", flush=True)
                continue

            fut = executor.submit(call_llm, idx, prompt)
            futures[fut] = idx

            if idx < n:
                time.sleep(sleep_s)

        # Recoger resultados según van llegando
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                _, raw = fut.result()
                results[idx] = raw
            except Exception as exc:
                print(f"[{idx:02d}] ERROR en llamada: {exc}", flush=True)
                results[idx] = ""

    # Guardar en orden
    ok = 0
    fail = 0
    for idx in sorted(results):
        raw = results[idx]
        out_dir = out_root / f"{idx:02d}"
        if not raw:
            fail += 1
            continue
        if save_candidate(out_dir, prompt, raw):
            print(f"[{idx:02d}] guardado -> {out_dir}", flush=True)
            ok += 1
        else:
            print(f"[{idx:02d}] fallo al parsear, raw en out.raw.txt", flush=True)
            fail += 1

    print(f"\nListo: {ok} ok, {fail} fallos -> {out_root}", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
