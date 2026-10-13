import json
from pathlib import Path

from hllm_pipeline.llm import prompt_gen, openai_generate, deepseek_generate
from hllm_pipeline.utils import read_text_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def split_hllm_output(result):
    payload = json.loads(result) if isinstance(result, str) else dict(result)
    code = payload.pop("code", "")
    return code, payload


def generate_hllm_heuristic(domain_route, instances_routes, llm_provider, llm_model, llm_thinking="medium", prompt_template=None):
    prompt_template = Path(prompt_template) if prompt_template else PROJECT_ROOT / "prompts" / "main_prompt_enhanced_02.md"

    domain_pddl = read_text_file(domain_route)
    representative_instances = "\n\n".join([f"## Representative instance {n+1}\n" + "```pddl\n" + read_text_file(path) + "```" for n, path in enumerate(instances_routes)])

    prompt = prompt_gen(prompt_template, {
        "domain_pddl": domain_pddl,
        "representative_instances": representative_instances,
        "heuristic_template": read_text_file(PROJECT_ROOT / "templates" / "heuristic_template.rs"),
    })

    match llm_provider:

        case "openai":
            result = openai_generate(prompt, model=llm_model, reasoning_effort=llm_thinking)

        case "deepseek":
            result = deepseek_generate(prompt, model=llm_model, reasoning_effort=llm_thinking)

        case _:
            raise NotImplementedError(f"Provider '{llm_provider}' is not supported.")

    code, explainability = split_hllm_output(result)
    return prompt, code, explainability


LLM_TEMPLATE_HEURISTIC_NAME = "llm_001"


def rename_heuristic(code, candidate_name):
    """Renombra llm_001 -> candidate_name en el código generado."""
    if candidate_name == LLM_TEMPLATE_HEURISTIC_NAME:
        return code
    return code.replace(LLM_TEMPLATE_HEURISTIC_NAME, candidate_name)


def generate_candidate(domain_route, instances_routes, llm_provider, llm_model, candidate_name, llm_thinking="medium"):
    prompt, code, explainability = generate_hllm_heuristic(
        domain_route, instances_routes, llm_provider, llm_model, llm_thinking=llm_thinking
    )
    code = rename_heuristic(code, candidate_name)

    return {
        "name": candidate_name,
        "prompt": prompt,
        "code": code,
        "explainability": explainability,
    }


def generate_candidates(domain_route, instances_routes, llm_provider, llm_model, num_candidates, llm_thinking="medium", name_prefix="h", start_index=1):
    """Genera num_candidates candidatos independientes (h_001, h_002, ...)."""
    candidates = []

    for offset in range(num_candidates):
        index = start_index + offset
        candidate_name = f"{name_prefix}_{index:03d}"

        candidates.append(
            generate_candidate(
                domain_route,
                instances_routes,
                llm_provider,
                llm_model,
                candidate_name,
                llm_thinking=llm_thinking,
            )
        )

    return candidates



