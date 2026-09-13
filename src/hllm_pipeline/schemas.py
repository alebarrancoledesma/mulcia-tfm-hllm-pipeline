def hllm_schema():

    return {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Rust source file implementing the heuristic. Must keep the name 'llm_001' unchanged.",
            },
            "what": {
                "type": "string",
                "description": "Concise technical explanation of what the heuristic does, what information it uses, any precomputation, and how the value is computed.",
            },
            "why": {
                "type": "string",
                "description": "Concise explanation of why this heuristic design was selected, including CVRP structural properties and patterns from the representative instances.",
            },
        },
        "required": ["code", "what", "why"],
        "additionalProperties": False,
    }
