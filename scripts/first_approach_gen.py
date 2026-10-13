"""Genera una tanda first_approach (4 prompts × 2 modelos).

Mismas instancias (P-n16-k8, P-n20-k2, P-n23-k8), mismos modelos/effort y mismo layout.
Solo genera (no compila). Salta lo que ya exista.

Uso:
    uv run python -u scripts/first_approach_gen.py first_approach_02
"""

import sys
from pathlib import Path

from hllm_pipeline.data import domain_path, instance_path
from hllm_pipeline.generate import generate_hllm_heuristic
from hllm_pipeline.utils import write_file, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_NAME = sys.argv[1] if len(sys.argv) > 1 else "first_approach_02"
OUT_ROOT = PROJECT_ROOT / "experiments" / EXP_NAME

PROMPT_NAMES = [
    "main_prompt",
    "main_prompt_enhanced_00",
    "main_prompt_enhanced_01",
    "main_prompt_enhanced_02",
]

RUNS = [
    ("openai", "gpt-5.6-sol"),
    ("deepseek", "deepseek-v4-pro"),
]

THINKING = "medium"
INSTANCE_IDS = ["P-n16-k8", "P-n20-k2", "P-n23-k8"]


def stem(provider, model):
    return f"{provider} - {model} - {THINKING}"


def already_done(out_dir, label):
    return all(
        (out_dir / name).exists()
        for name in (f"out - {label}.rs", f"out - {label}.json", f"prompt - {label}.md")
    )


def main():
    domain = domain_path()
    instances = [instance_path("P", iid) for iid in INSTANCE_IDS]
    for path in [domain, *instances]:
        if not path.exists():
            raise FileNotFoundError(path)

    failures = []
    jobs = [(p, prov, model) for p in PROMPT_NAMES for prov, model in RUNS]
    print(f"Generando {len(jobs)} candidatos -> {OUT_ROOT}", flush=True)

    for i, (prompt_name, provider, model) in enumerate(jobs, start=1):
        label = stem(provider, model)
        out_dir = OUT_ROOT / prompt_name
        tag = f"{prompt_name} / {label}"

        if already_done(out_dir, label):
            print(f"[{i}/{len(jobs)}] SKIP (ya existe) {tag}", flush=True)
            continue

        print(f"[{i}/{len(jobs)}] {tag} ...", flush=True)
        try:
            prompt, code, explainability = generate_hllm_heuristic(
                domain,
                instances,
                provider,
                model,
                llm_thinking=THINKING,
                prompt_template=PROJECT_ROOT / "prompts" / f"{prompt_name}.md",
            )
        except Exception as exc:
            print(f"  FALLO: {exc}", flush=True)
            failures.append((tag, str(exc)))
            continue

        write_file(out_dir / f"prompt - {label}.md", prompt, overwrite=False)
        write_file(out_dir / f"out - {label}.rs", code, overwrite=False)
        write_json(out_dir / f"out - {label}.json", explainability, overwrite=False)
        print(f"  OK ({len(code)} chars)", flush=True)

    if failures:
        print("\nFallos:", flush=True)
        for tag, err in failures:
            print(f"  - {tag}: {err}", flush=True)
        raise SystemExit(1)

    print(f"\nListo: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
