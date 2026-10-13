from hllm_pipeline.generate import generate_hllm_heuristic
from hllm_pipeline.utils import write_file, write_json


# prompt, code, explainability = generate_hllm_heuristic(r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\data\processed\pddl\domain.pddl", [r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\data\processed\pddl\P\P-n16-k8.pddl", r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\data\processed\pddl\P\P-n20-k2.pddl", r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\data\processed\pddl\P\P-n23-k8.pddl"], "openai", "gpt-5.6-sol", llm_thinking = "medium")
prompt, code, explainability = generate_hllm_heuristic(r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\data\processed\pddl\domain.pddl", [r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\data\processed\pddl\P\P-n16-k8.pddl", r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\data\processed\pddl\P\P-n20-k2.pddl", r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\data\processed\pddl\P\P-n23-k8.pddl"], "deepseek", "deepseek-v4-pro", llm_thinking = "medium")

write_file(r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\prompt.md", prompt)
write_file(r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\out.rs", code)
write_json(r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\out.json", explainability)
