"""Generate real GRASP candidates: prompt.md, original out.py, and what/why out.json."""
import argparse
import ast
import concurrent.futures
import json
from pathlib import Path
import sys
import threading
import time

from dotenv import load_dotenv
from hllm_pipeline.llm import openai_generate, prompt_gen

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {"E": ["E-n23-k3", "E-n33-k4", "E-n51-k5", "E-n76-k14", "E-n101-k8"],
            "P": ["P-n16-k8", "P-n23-k8", "P-n50-k10", "P-n76-k4", "P-n101-k4"]}
SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in ("code", "what", "why")},
          "required": ["code", "what", "why"], "additionalProperties": False}


def build_prompt(family):
    examples = []
    for name in EXAMPLES[family]:
        path = ROOT.parent / "mulcia-tfm-artifacts" / "data" / "cvrplib" / family / f"{name}.vrp"
        text = path.read_text(encoding="utf-8")
        # Omit only COMMENT fields revealing benchmark objective values.
        text = "".join(line for line in text.splitlines(keepends=True)
                       if not (line.lstrip().startswith("COMMENT")
                               and any(word in line.lower() for word in ("optimal", "best value"))))
        if not all(section in text for section in
                   ("EUC_2D", "CAPACITY", "NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF")):
            raise ValueError("Incomplete Euclidean VRP example")
        examples.append(f"## {name}\n```text\n{text}\n```")
    return prompt_gen(ROOT / "prompts" / "grasp_rcl.md", {
        "family": family, "representative_instances": "\n\n".join(examples),
        "heuristic_template": (ROOT / "templates" / "grasp_heuristic_template.py").read_text(encoding="utf-8")})


def validate(code):
    """Check syntax and exact keyword-callable signature, without import/execution."""
    try:
        tree = ast.parse(code)
        compile(tree, "<generated>", "exec")
    except (SyntaxError, ValueError, TypeError):
        return False, False
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "heuristic"]
    if len(funcs) != 1:
        return True, False
    args = funcs[0].args
    signature_ok = ([arg.arg for arg in args.args] ==
                    ["instance", "solution", "unassigned", "route_index", "customer", "position"]
                    and not (args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults))
    return True, signature_ok


def save_candidate(folder, raw):
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or set(payload) != {"code", "what", "why"}
            or not all(isinstance(value, str) for value in payload.values()) or not payload["code"].strip()):
        raise ValueError("Invalid response schema")
    # Preserve the code exactly as returned: no repairs or Markdown stripping.
    (folder / "out.py").write_text(payload["code"], encoding="utf-8", newline="\n")
    (folder / "out.json").write_text(
        json.dumps({key: payload[key] for key in ("what", "why")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    syntax, signature = validate(payload["code"])
    print(f"[{folder.name}] saved; syntax={syntax}, signature={signature}", flush=True)
    return syntax and signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["E", "P"], required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--rpm", type=float, default=6)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n < 1 or args.workers < 1 or not 0 < args.rpm <= 60:
        parser.error("n/workers must be positive; rpm must be in (0, 60]")
    started = time.monotonic()
    load_dotenv(ROOT / ".env", override=False)
    prompt = build_prompt(args.family)
    root = ROOT / "experiments" / f"grasp-{args.family}"
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error("Output directory must be empty for a fresh batch")
    pending = list(range(1, args.n + 1))
    preserved = 0
    if args.resume and root.exists():
        expected = {f"{idx:02d}" for idx in pending}
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.name not in expected:
                parser.error(f"Unexpected output entry: {folder.name}")
            prompt_path = folder / "prompt.md"
            if not prompt_path.is_file() or prompt_path.read_bytes() != prompt.encode("utf-8"):
                parser.error(f"Prompt mismatch or missing: {folder.name}; outputs preserved")
            files = {path.name for path in folder.iterdir()}
            if files == {"prompt.md"}:
                continue
            if files != {"prompt.md", "out.py", "out.json"}:
                parser.error(f"Partial/unexpected outputs: {folder.name}; outputs preserved")
            try:
                metadata = json.loads((folder / "out.json").read_text(encoding="utf-8"))
                valid = (isinstance(metadata, dict) and set(metadata) == {"what", "why"}
                         and all(isinstance(value, str) for value in metadata.values())
                         and all(validate((folder / "out.py").read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                valid = False
            if not valid:
                parser.error(f"Invalid complete candidate: {folder.name}; outputs preserved")
            pending.remove(int(folder.name))
            preserved += 1
    root.mkdir(exist_ok=True)
    print(f"Preserved={preserved}; pending={len(pending)}", flush=True)
    print(f"Family={args.family}; examples={EXAMPLES[args.family]}; n={args.n}\n"
          f"Provider=openai; model={args.model}; reasoning=medium; strict_schema=True\n"
          f"workers={args.workers}; rpm={args.rpm}; max_retries=0; timeout=300s\n"
          f"Output={root}", flush=True)
    stop = threading.Event()
    lock = threading.Lock()
    calls = 0
    next_call = 0.0

    def call_llm(idx):
        nonlocal calls, next_call
        with lock:
            if stop.wait(max(0, next_call - time.monotonic())):
                return None
            folder = root / f"{idx:02d}"
            folder.mkdir(exist_ok=args.resume)
            if not (folder / "prompt.md").exists():
                (folder / "prompt.md").write_text(prompt, encoding="utf-8", newline="\n")
            calls += 1
            next_call = time.monotonic() + 60 / args.rpm
            print(f"[{idx:02d}] sending real provider call #{calls}", flush=True)
        try:
            raw = openai_generate(prompt, model=args.model, reasoning_effort="medium",
                                  response_schema=SCHEMA, raise_errors=True, max_retries=0, timeout=300)
        except Exception as exc:
            stop.set()
            status = getattr(exc, "status_code", None)
            # Do not print exception bodies, headers, URLs, or credentials.
            print(f"[{idx:02d}] API ERROR; http_status={status if isinstance(status, int) else 'unavailable'}; "
                  "batch stopped, no retries", flush=True)
            return False
        print(f"[{idx:02d}] response received ({len(raw)} chars)", flush=True)
        try:
            return save_candidate(folder, raw)
        except (ValueError, TypeError, OSError):
            stop.set()
            print(f"[{idx:02d}] ERROR saving/decoding response; batch stopped", flush=True)
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(call_llm, idx) for idx in pending]
        results = [job.result() for job in concurrent.futures.as_completed(jobs)]
    ok = results.count(True)
    failed = results.count(False)
    print(f"Family={args.family}: preserved={preserved}, provider_calls={calls}, successes={ok}, errors={failed}, "
          f"not_requested={results.count(None)}; elapsed_seconds={time.monotonic() - started:.1f}; "
          "evaluation not started", flush=True)
    return 0 if preserved + ok == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
