"""Ejemplo: genera N candidatos para una familia (solo PROMPT, sin PlanForge)."""

from hllm_pipeline import experiment

config = experiment.build_config(
    family="P",
    provider="deepseek",
    model="deepseek-v4-pro",
    num_candidates=2,
    reasoning_effort="medium",
)

result = experiment.run_experiment("exp_P_001", config)

print(f"Directorio del experimento: {result['exp_dir']}")
print(f"Candidatos: {result['candidates']}")
