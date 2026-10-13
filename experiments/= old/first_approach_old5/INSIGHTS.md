# first_approach

8 candidatos: `main` / `_00` / `_01` / `_02` × OpenAI `gpt-5.6-sol` y DeepSeek `deepseek-v4-pro` (reasoning `medium`). El LLM vio `P-n16-k8`, `P-n20-k2`, `P-n23-k8`. GBFS 300s sobre esas mismas tres, más `blind` / `ff` / `lmcutnumeric`.

Prompts de esta tanda: `capacity`/`demand`/`max_vehicles` por nombre exacto, y si no están, `new()` sigue con default permisivo. No abortar.

CSV: `metrics.csv`. Logs: `logs/`. DeepSeek solo tiene `main` en n16 (completo) y un n20 cortado; `_00`/`_01`/`_02` DeepSeek no se llegaron a correr.

---

## Qué mira cada fila cualitativa

| Fila | Qué |
|---|---|
| Tiempo / coste / tokens | Llamada al LLM, no PlanForge. Tokens con `tiktoken`. Coste aprox. (reasoning OpenAI no entra). |
| Calidad | Si el `.rs` parece una heurística CVRP usable. No es la media del resto. |
| Alineación | Si cumple lo que pedía *esa* plantilla. |
| Client-based | Flota, demandas, via-depot; sin copiar literales de las P. |
| Operadores | Distancias desde `metric_operator_cost_from_initial_values`. |
| CVRP | Load, capacidad/demanda, flota, via-depot, vuelta al depot. |
| Multi-comp. | `h` como suma de términos (`h_routing + h_capacity`…), no un solo número. |
| Estado | `h` cambia con loc, visitados, load, vehicles. |
| Robustez | `new()` si falta un fluent: `Ok` + fallback vs abortar. |
| JSON | Si `what`/`why` (y `design` en `_01`/`_02`) describen el código. |
| Riesgo cargo | Pinta del `.rs`. Más = peor. |

Notas 1–10: más = mejor, salvo riesgo cargo.

## Diseño (lectura del `.rs` / JSON)

| | `main` OA | `main` DS | `_00` OA | `_00` DS | `_01` OA | `_01` DS | `_02` OA | `_02` DS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiempo gen. | 2m 25s | 6m 36s | 1m 45s | 8m 01s | 1m 55s | 5m 20s | 3m 36s | 7m 11s |
| Coste | ~$0.10 | ~$0.03 | ~$0.10 | ~$0.03 | ~$0.12 | ~$0.03 | ~$0.13 | ~$0.04 |
| Tokens in/out | 37k / 4.3k | 37k / 4.1k | 37k / 3.9k | 37k / 3.6k | 39k / 5.2k | 39k / 4.7k | 39k / 5.3k | 39k / 5.7k |
| Calidad | 6.5 | 7.5 | 6.5 | 7 | **8** | 7 | **8** | 7.5 |
| Alineación | 6 | 6 | 6 | 6 | 8 | 7 | **9** | 8 |
| Client-based | 6 | 7 | 6 | 7 | 7 | 7 | **8** | **8** |
| Operadores | 8 | 8 | 8 | 8 | **9** | 8 | **9** | **9** |
| CVRP | 6 | **8** | 7 | 8 | **8** | 8 | **8** | 8 |
| Multi-comp. | 6 | 3 | 3 | 3 | **9** | 4 | **9** | 5 |
| Estado | 8 | **9** | 8 | 8 | 8 | **9** | 8 | **9** |
| Robustez | **8** | **8** | **8** | **8** | **8** | **8** | **8** | **8** |
| JSON | 6 | 6 | 6 | 6 | **9** | 7 | **9** | 8 |
| Riesgo cargo | 2 | 2 | 2 | 3 | 2 | 3 | 2 | **6** |

| Prompt | OpenAI | DeepSeek |
|---|---|---|
| `main` | 75% greedy + 25% MST. Capacidad solo si el PNE sobrevive. | Greedy capacitado (direct / via-depot / return). Compiló. |
| `_00` | Rollout NN con load. Un solo `h`. | Igual de idea: completion greedy. |
| `_01` | `h_routing + h_capacity + h_vehicles`. First-transition + outgoing. | Greedy + chequeo de flota. El JSON habla de tres términos; el `return` es solo `h_routing`. |
| `_02` | Como `_01` más `instance_patterns`. | Greedy + penalty de recursos + patterns. `fill_state().map_err` — `fill_state` devuelve `()`, no compila. |

| # | Quién | Por qué (solo diseño) |
|---|---|---|
| 1 | `_02` OpenAI | Tres componentes, patterns, `new()` no aborta |
| 2 | `_01` OpenAI | Igual de sólido, sin el extra de `_02` |
| 3 | `main` DeepSeek | Greedy CVRP entero; más simple, pero coherente |
| 4 | `_02` DeepSeek | Bien planteado; el `map_err` lo deja fuera |
| 5 | `_01` DeepSeek | Greedy bueno; el JSON promete más de lo que suma |
| 6 | `_00` DeepSeek | Completion capacitado, un término |
| 7 | `_00` OpenAI | NN usable, poco desglose |
| 8 | `main` OpenAI | Blend geométrico; capacidad opcional |

Robustez alta en los 8: nadie aborta `new()` por `capacity()`. Eso es el cambio del prompt.

---

## Métricas (CSV + logs)

GBFS 300s. Si el planner no imprime `Search time` / `Expanded` (timeout o corte), el CSV deja huecos; esos números salen del `.txt`.

### Qué se llegó a correr

| Candidato | Compila | Búsqueda |
|---|---|---|
| blind, ff, lmcutnumeric | sí | sí |
| `main` OpenAI | sí | n16, n20, n23 |
| `_00` OpenAI | sí | n16, n20, n23 |
| `_01` OpenAI | sí | n16, n20, n23 |
| `_02` OpenAI | sí | n16, n20, n23 |
| `main` DeepSeek | sí | n16 y n20 con `h=0` (ciega); n23 no |
| `_00` / `_01` DeepSeek | no corrido | — |
| `_02` DeepSeek | no (el `.rs` tiene `fill_state().map_err`) | — |

Las cuatro OpenAI construyen y buscan. En old4, `_01`/`_02` OA morían en `new()` por `capacity()`.

### Resultados

**P-n16-k8**

| | solved | t (s) | exp | cost |
|---|---|---:|---:|---:|
| `main` DeepSeek | sí | 7.90 | 2.73M | 450 (`h=0`, no guía) |
| blind | sí | 13.80 | 5.22M | 468 |
| `_01` OpenAI | sí | **0.55** | **449k** | 490 |
| `main` OpenAI | sí | 2.46 | 1.40M | 537 |
| `_00` OpenAI | sí | 1.31 | 894k | 537 |
| ff | sí | 0.42 | 81k | 540 |
| `_02` OpenAI | sí | 1.94 | 1.25M | 548 |
| lmcutnumeric | no | ~300 | ~192k | — |

**P-n20-k2**

| | solved | t (s) | exp | cost |
|---|---|---:|---:|---:|
| `main` OpenAI | sí | 11.47 | 4.25M | **326** |
| `_00` OpenAI | sí | **5.90** | 2.96M | **326** |
| `_02` OpenAI | sí | 20.60 | 10.05M | 431 |
| `_01` OpenAI | sí | 21.24 | 10.10M | 451 |
| ff | sí | 6.63 | 1.41M | 529 |
| blind | no | 300 | 41.0M | — |
| lmcutnumeric | no | ~300 | ~70k | — |
| `main` DeepSeek | no | cortado | — | `h=0` siempre |

**P-n23-k8** — nadie resuelve. Del log, último `h` y expansiones al corte:

| | último h | exp (aprox.) |
|---|---:|---:|
| `_01` OpenAI | **8** | 10.0M |
| `_02` OpenAI | 29 | 59.1M |
| ff | 39 | 24.9M |
| `main` OpenAI | 42 | 45.3M |
| `_00` OpenAI | 42 | 33.3M |
| blind | — | 26.6M |
| lmcutnumeric | 69 | ~451 |

### Lectura rápida de los números

**n16.** DeepSeek `main` saca 450, pero el log dice `Initial heuristic value: 0` y no vuelve a bajar `h`: GBFS va a ciegas. No cuenta como heurística. `_01` OA es el LLM que de verdad guía (0.55s, 449k, coste 490). FF es más rápido (0.42s) con plan 540. `_02` OA es el peor coste de los que resuelven (548).

**n20.** Las dos geométricas (`main` / `_00` OA) empatan a 326 y ganan a FF (529). `_00` OA lo hace en 5.9s; `main` OA tarda el doble. `_01`/`_02` OA resuelven más caros (451/431) y expanden ~10M. DeepSeek `main` otra vez `h=0`; se corta a los 5 min (~14 GB).

**n23.** Nadie cierra. `_01` OA baja `h` hasta 8; el resto se queda en 29–42. Guía más, no le da el tiempo.

**lmcutnumeric.** Casi no mueve. Cada eval es cara.

---

## Conclusiones

El arreglo del prompt funcionó: las OpenAI `_01`/`_02` ya no pegan un tiro en `new()` y corren.

En papel ganan `_01`/`_02` OA (tres componentes). En máquina, en n16/n20, las más simples (`main`/`_00` OA) dan mejor coste. `_01` OA se justifica en n16 (rápida, 490) y sobre todo en n23 (`h=8`). `_02` OA es la que más se parece al prompt y la que peor coste tiene donde sí resuelve.

DeepSeek `main` no guía: `h=0` desde el estado inicial (lee `visited` mal, ver abajo). El 450 de n16 es un plan a ciegas. El resto de DeepSeek no está en el CSV; `_02` DS además no va a compilar.

**Por qué DeepSeek da 0.** En `main` DS, `parse_fact_name` acepta `Atom visited(c)` y `NegatedAtom visited(c)`. Se queda con el último valor del dominio (el negado) y además usa **una sola** `visited_var` para todos los clientes. En el estado inicial eso marca a todos como visitados, está en el depot, y `compute_heuristic` hace `return Ok(0.0)`. OpenAI solo parsea `Atom visited(` y guarda `(var, value)` por cliente; por eso arranca en 160–210 y va bajando.

Nadie saca n23 en 5 min. FF cubre n16+n20 con planes caros. Blind solo n16. Lo que corre de LLM cubre lo mismo que FF y, en n20, con planes más cortos.
