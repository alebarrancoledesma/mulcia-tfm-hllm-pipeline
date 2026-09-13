import json
from pathlib import Path

from hllm_pipeline.llm import prompt_gen, openai_generate, deepseek_generate
from hllm_pipeline.utils import read_text_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def split_hllm_output(result):
    payload = json.loads(result) if isinstance(result, str) else dict(result)
    code = payload.pop("code", "")
    return code, payload


def generate_hllm_heuristic(domain_route, instances_routes, llm_provider, llm_model, llm_thinking = "medium"):

    domain_pddl = read_text_file(domain_route)
    representative_instances = "\n\n".join([f"## Representative instance {n+1}\n" + "```pddl\n" + read_text_file(path) + "```" for n, path in enumerate(instances_routes)])

    prompt = prompt_gen(PROJECT_ROOT / "prompts" / "main_prompt_enhanced_02.md", {
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



