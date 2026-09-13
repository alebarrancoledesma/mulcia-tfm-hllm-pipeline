from hllm_pipeline.generate import generate_hllm_heuristic
from hllm_pipeline.utils import write_file, write_json


# prompt, code, explainability = generate_hllm_heuristic(r"C:\Code\GitHub\TFM\data\cvrplib\domain.pddl", [r"C:\Code\GitHub\TFM\data\cvrplib\A\A-n32-k5.pddl", r"C:\Code\GitHub\TFM\data\cvrplib\A\A-n37-k5.pddl"], "openai", "gpt-5.6-sol", llm_thinking = "medium")
prompt, code, explainability = generate_hllm_heuristic(r"C:\Code\GitHub\TFM\data\cvrplib\domain.pddl", [r"C:\Code\GitHub\TFM\data\cvrplib\A\A-n32-k5.pddl", r"C:\Code\GitHub\TFM\data\cvrplib\A\A-n37-k5.pddl"], "deepseek", "deepseek-v4-pro", llm_thinking = "medium")

write_file(r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\prompt.md", prompt)
write_file(r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\out.rs", code)
write_json(r"C:\Code\GitHub\TFM\mulcia-tfm-hllm-pipeline\out.json", explainability)
