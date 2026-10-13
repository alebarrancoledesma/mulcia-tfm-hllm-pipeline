# hllm-pipeline

Heurísticas `h(s)` en Rust generadas por LLM para PlanForge, sobre el dominio CVRP (PDDL numérico). Bloque 1 del TFM: pipeline experimental con múltiples candidatos, selección en DEV y evaluación final en TEST.

## Visión general

Dos familias de cliente CVRP, **P** y **E** (Augerat **P** y Christofides–Eilon **E** — mismo `domain.pddl`, distinta procedencia). Se usan solo las instancias **pequeñas** (n≤23 en P, n≤33 en E) porque A/B Augerat y los P/E grandes no son asequibles para GBFS/FF a pocos minutos, y además P grande está cruzado con E grande. Cada familia se divide en tres roles disjuntos:

| Split | Quién lo ve | Para qué |
|---|---|---|
| `prompt` | el LLM | instancias representativas embebidas en el prompt (k=3: pequeña / mediana / grande-compleja) |
| `dev` | solo `rank.py` | seleccionar/rankear candidatos |
| `test` | nadie hasta el final | evaluación final, congelada |

**Anti-leakage:** el LLM solo ve `prompt`; el ranking solo ve `dev`; `test` se toca una única vez, con el candidato ya congelado.

Objetivo final: matriz de transferencia 2×2 (+ genérica):

```
        P-test   E-test
h_P       ✓        ✓
h_E       ✓        ✓
h_G       ✓        ✓
```

## Estructura del repositorio

```
mulcia-tfm-hllm-pipeline/
├── data/
│   ├── processed/pddl/
│   │   ├── domain.pddl          # dominio PlanForge CVRP compartido
│   │   ├── P/*.pddl             # set Augerat P
│   │   └── E/*.pddl             # set Christofides–Eilon
│   └── splits/
│       ├── P/{prompt,dev,test}.json
│       └── E/{prompt,dev,test}.json
│
├── prompts/
│   └── main_prompt_enhanced_02.md     # plantilla de prompt en uso
├── templates/
│   └── heuristic_template.rs          # esqueleto Rust que recibe el LLM
│
├── src/hllm_pipeline/
│   ├── llm.py          # llamadas a OpenAI/DeepSeek, prompt templating, pricing
│   ├── schemas.py      # JSON schema de la salida hllm
│   ├── utils.py        # lectura/escritura de texto y json (sin overwrite)
│   ├── data.py         # familia/split -> ruta PDDL
│   ├── generate.py     # llamada al LLM + generación de candidatos
│   ├── benchmark.py    # compila + ejecuta un candidato vía PlanForge
│   ├── rank.py         # rankea un metrics.csv de DEV (nunca TEST)
│   └── experiment.py   # orquesta config -> generar -> benchmark -> rankear -> test
│
├── experiments/
│   └── <exp_name>/                    # un registro autocontenido por ejecución
│
├── results/first_approach/            # ejecuciones ad-hoc previas + INSIGHTS.md
└── scripts/
    ├── gen_candidates.py   # N candidatos en paralelo (OpenAI + main_prompt)
    ├── sandbox.py          # script original de generación
    └── run_experiment.py   # ejemplo de uso de experiment.py
```

### Splits actuales (k=3)

```
P/prompt: P-n16-k8, P-n20-k2, P-n23-k8
P/dev:    P-n16-k8, P-n19-k2, P-n20-k2, P-n22-k8
P/test:   P-n21-k2, P-n22-k2

E/prompt: E-n13-k4, E-n22-k4, E-n30-k3
E/dev:    E-n13-k4, E-n22-k4, E-n23-k3, E-n30-k3
E/test:   E-n31-k7, E-n33-k4
```

Detalle y genealogía: `data/README.md`. `k` = `len(prompt_instance_ids)`, se cambia editando `data/splits/<familia>/prompt.json` o pasando `prompt_instance_ids` en la config.

## Experimentos

`experiment.new_experiment_dir(...)` crea `experiments/<exp_name>/` (lanza excepción si ya existe):

```
experiments/exp_A_001/
├── config.json          # family, provider, model, reasoning_effort, ids prompt/dev/test, ...
├── candidates/
│   ├── h_001.rs          # código Rust
│   ├── h_001.json        # what/why/instance_patterns/design/...
│   └── ...
├── prompts/
│   └── h_001.md          # prompt exacto enviado al LLM
├── dev/metrics.csv        # resultados por (candidato, instancia) en DEV
├── test/metrics_<family>.csv
└── ranking.json          # ranking + excluidos + candidato seleccionado
```

## Puesta en marcha

```powershell
cd mulcia-tfm-hllm-pipeline
uv sync
```

Requiere `.env` con `OPENAI_KEY` y/o `DEEPSEEK_API_KEY` (ya presente, no commiteado).

## Lanzar un experimento

### 1. Generar candidatos

`scripts/gen_candidates.py`: OpenAI `gpt-5.6-sol` (reasoning medium), plantilla `prompts/main_prompt.md`, N llamadas en paralelo con sleep anti-rate-limit. Las instancias salen del JSON de `--split` (`family` + `instances`).

```powershell
# familia P (default: data/splits/P/prompt.json)
uv run python -u scripts/gen_candidates.py experiment-P --split P/prompt

# familia E
uv run python -u scripts/gen_candidates.py experiment-E --split E/prompt
```

También vale la ruta completa (`--split data/splits/E/prompt.json`). Otros flags: `--n 10` (default), `--rpm 20` (segundos entre envíos = 60/rpm).

Salida: `experiments/<nombre>/01/` … `10/` con `out.rs`, `out.json`, `prompt.md`. Si `01/out.rs` ya existe, ese índice se salta.

Evaluar después con `bash compile_and_run.sh` en esa carpeta (CSV + `logs/`). Rankear con el apartado 2.

### 2. Rankear un `metrics.csv`

Lee el CSV de DEV (`candidate,instance,solved,runtime,...`) y escribe `ranking.csv` al lado. Criterio: coverage → menor gap relativo al mejor coste observado → agile score → menos expansiones.

```powershell
uv run python -m hllm_pipeline.rank experiments/experiment001
```

También vale la ruta al CSV:

```powershell
uv run python -m hllm_pipeline.rank experiments/experiment001/metrics.csv
```

Salida por defecto: `experiments/experiment001/ranking.csv`. Otro destino: `-o ruta/ranking.csv`.

Desde Python: `from hllm_pipeline.rank import rank_candidates` y `rank_candidates("experiments/experiment001")`.

### 3. Evaluar en TEST (solo una vez, candidato ya congelado)

```python
test_records = experiment.run_test_evaluation(result["exp_dir"], config, ranking["selected"])
```

Para transferencia P↔E: `test_family="E"` (o "P").

O de punta a punta: `experiment.run_experiment("exp_A_003", config, do_dev_benchmark=True, do_test=True)`.

## Limitaciones actuales

- `benchmark.py` está basado en el CLI real de PlanForge pero aún no se ha probado con un `cargo build` real.
- El crate runner es único y compartido: compilar/ejecutar candidatos es secuencial, no paralelo.
- `h_G` y la matriz de transferencia completa no están conectadas todavía; `test_family` en `run_test_evaluation` es el gancho para eso.
