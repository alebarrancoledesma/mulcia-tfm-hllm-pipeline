# first_approach

8 heurísticas (`main_prompt` / `_00` / `_01` / `_02` × OpenAI `gpt-5.6-sol` y DeepSeek `deepseek-v4-pro`, reasoning `medium`). El LLM vio `P-n16-k8`, `P-n20-k2` y `P-n23-k8`. GBFS 300s sobre esas mismas tres, más `blind`, `ff` y `lmcutnumeric`. CSV: `metrics.csv`. Logs: `logs/`.

La tabla de abajo es solo lectura del `.rs` y del JSON (`what`/`why`/etc.). Las notas 1–10: más = mejor, salvo riesgo de compilación (más = peor). Coste y tiempo de generación: `tiktoken` + mtimes del 13/09/2026 (~44 min, ~$0.59 visibles; el reasoning de OpenAI no entra).

## Qué es cada fila

| Fila | Qué mira |
|---|---|
| Tiempo / coste / tokens | Medidas de la llamada al LLM, no de PlanForge. |
| Calidad global | Si el diseño parece una heurística CVRP usable. No es la media del resto. |
| Alineación prompt | Si cumple lo que pedía esa plantilla (MUST, campos JSON, no tour fijo en `_02`). |
| Client-based | Si usa estructura de las P (flota, demandas, via-depot) sin copiar literales. |
| Costes vía operadores | Si saca distancias de `metric_operator_cost_from_initial_values`, no solo PNEs. |
| CVRP completo | Load, capacidad/demanda, flota, via-depot y vuelta al depot en el código. |
| Multi-componente | Si `h` es suma de términos reales (`h_routing + h_capacity`…), no un solo número. |
| State-dependent | Si `h` cambia con ubicación, visitados, carga, vehículos. |
| Robustez | Qué hace `new()` si un fluent no sale por nombre (`Ok` + fallback vs abortar). |
| Interpretabilidad | Si el JSON describe el código (`what`/`why`; `_01`/`_02` también `design`, etc.). |
| Riesgo compilación | Pinta del `.rs` antes de `cargo`. Luego se contrastó con los logs. |

## Tabla

| | `main` OA | `main` DS | `_00` OA | `_00` DS | `_01` OA | `_01` DS | `_02` OA | `_02` DS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiempo gen. | 2m 19s | 11m 28s | 1m 39s | 9m 54s | 2m 16s | 6m 40s | 2m 23s | 7m 01s |
| Coste | ~$0.10 | ~$0.03 | ~$0.10 | ~$0.03 | ~$0.13 | ~$0.03 | ~$0.13 | ~$0.04 |
| Tokens in/out | 36k / 3.6k | 36k / 4.3k | 37k / 3.3k | 37k / 6.1k | 38k / 6.4k | 38k / 5.5k | 39k / 6.4k | 39k / 7.1k |
| Calidad global | 6.5 | 7 | 5.5 | 7 | 7 | 7.5 | **7.5** | **8** |
| Alineación | 6 | 6 | 6 | 6 | 8 | 8 | **9** | 8 |
| Client-based | 5 | 6 | 5 | 6 | 7 | 7 | **8** | **8** |
| Operadores | 8 | 8 | 7 | 8 | **9** | 8 | **9** | **9** |
| CVRP completo | 4 | **8** | 3 | 8 | 8 | 8 | **9** | 8 |
| Multi-componente | 3 | 2 | 2 | 3 | 7 | **8** | **9** | 7 |
| State-dependent | 8 | **9** | 7 | 8 | 8 | **9** | 8 | **9** |
| Robustez | **8** | 7 | **8** | 7 | 3 | 6 | 3 | 7 |
| Interpretabilidad | 6 | 6 | 6 | 6 | **9** | 8 | **9** | 8 |
| Riesgo compilación | 2 | 3 | 2 | 3 | 4 | 3 | 4 | 3 |

## Por prompt

**`main_prompt`.** OA: blend 0.8 NN + 0.2 MST, sin capacidad. DS: greedy capacitado (direct / via-depot).

**`enhanced_00`.** OA: NN directed, solo `visit` + return; el `why` dice que no toca capacidad. DS: greedy multi-ruta; `capacity` opcional.

**`enhanced_01`.** Pedía componentes y JSON largo. OA: primera transición + mínimos salientes + `h_capacity`; `new()` exige `capacity()`/`demand` por nombre. DS: greedy + tres términos; `load`/`vehicles` obligatorios.

**`enhanced_02`.** Igual que `_01` más `instance_patterns`. OA: tres componentes; aborta si no encuentra `capacity()`. DS: greedy + `h_capacity`; si faltan demandas pone 1.0.

| # | Quién | Por qué |
|---|---|---|
| 1 | `_02` OpenAI | Tres componentes, patterns, lo que pedía `_02` |
| 2 | `_02` DeepSeek | Greedy + patterns; el `return` solo suma dos términos |
| 3 | `_01` DeepSeek | Greedy + tres términos, sin patterns |
| 4 | `_01` OpenAI | First-transition + capacidad, sin flota |
| 5 | `main` DeepSeek | Greedy CVRP, JSON corto |
| 6 | `_00` DeepSeek | Multi-ruta, sin descomponer `h` |
| 7 | `main` OpenAI | NN+MST, sin capacidad |
| 8 | `_00` OpenAI | NN puro |

---

## Análisis métricas

GBFS, 300s. Instancias del prompt. El CSV deja runtime/expanded vacíos cuando el planner no imprime `Search time:` / `Expanded` (timeout o error en `new()`). Esos casos salen de los `.txt`.

### Quién llegó a buscar

| Candidato | Compila | Qué pasó al buscar |
|---|---|---|
| `blind`, `ff`, `lmcutnumeric` | sí (`planforge`) | Corren. |
| `main` / `_00` OpenAI | sí | Buscan. |
| `_01` OpenAI | sí | `new()`: *recovered invalid capacity or max_vehicles values* |
| `_02` OpenAI | sí | `new()`: *could not recover capacity()* |
| Los 4 DeepSeek | no | `main` y `_02`: `fill_state(...).map_err` sobre `()`. `_00` y `_01`: `for op in &ops` (`&&Vec` no itera). |

`_01`/`_02` OpenAI hacen exactamente lo que el prompt decía que no: mirar `capacity()` por nombre. En estas P no está (o el valor no es usable). No hay búsqueda.

### Quién resolvió

| | P-n16-k8 | P-n20-k2 | P-n23-k8 |
|---|---|---|---|
| blind | sí, 13.8s, cost 468, 5.2M exp | timeout (~41M exp) | timeout (~27M exp) |
| ff | sí, 0.42s, cost 540, 81k exp | sí, 6.6s, cost 529, 1.4M exp | timeout (~25M exp, h=39) |
| lmcutnumeric | timeout (~192k exp, h=38) | timeout (~70k exp, h=77) | timeout (~451 exp, h=69) — muy lenta |
| `main` OA | sí, 1.76s, cost 511, 1.05M exp | sí, 8.7s, cost **296**, 3.3M exp | timeout (~82M exp, h=45) |
| `_00` OA | sí, 1.30s, cost 537, 894k exp | sí, 5.9s, cost 326, 3.0M exp | timeout (~33M exp, h=42) |
| resto LLM | no corre | no corre | no corre |

Nadie saca P-n23 en 5 min.

### Números que importan

En **n16** blind encuentra el plan más barato (468). FF es el más rápido (0.42s) pero el plan es el más caro (540). `main` OA se queda en medio (511, 1.8s). `_00` OA casi como FF en coste (537) y más lento.

En **n20** las dos OpenAI que corren ganan a FF en coste: 296 y 326 contra 529. Tardan parecido (6–9s vs 6.6s) y expanden más (~3M vs 1.4M). GBFS no es óptimo; aquí la geometría empuja a planes más cortos que FF.

En **n23** todos timeout. FF y las OpenAI geométricas sí avanzan (millones de expansiones, h bajando). lmcutnumeric casi no mueve (451 exp en n23): cada evaluación es cara.

### Encaje con la tabla cualitativa

El ranking de diseño ponía `_02` OA el 1. En máquina es el que menos sirve: no construye. Las que puntúan bajo en “CVRP completo” (`main` / `_00` OA) son las únicas LLM que resuelven algo. DeepSeek tenía mejor pinta de CVRP y no pasa de `cargo`.

Resumen: en este run, lo que corre y resuelve es geometría OpenAI + FF. Lo “más CVRP” del papel no llega a `compute_heuristic`.
