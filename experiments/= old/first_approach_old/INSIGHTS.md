# INSIGHTS — First Approach (HLLM Heuristic Generation)

Análisis cualitativo de las 8 ejecuciones del pipeline sobre instancias representativas CVRPLib (`A-n32-k5`, `A-n37-k5`), dominio PlanForge CVRP, modelos **gpt-5.6-sol** (OpenAI) y **deepseek-v4-pro** (DeepSeek), reasoning `medium`.

**Nota:** estas valoraciones se basan en revisión estática del JSON generado (`code`, `what`, `why`, metadatos). No incluyen aún medidas empíricas en PlanForge (compilación, nodos expandidos, tiempo).

**Escala usada en la tabla:** 1–10 (mayor = mejor), excepto *Riesgo compilación* (mayor = peor).

**Coste aprox.:** estimado con `tiktoken` sobre el prompt materializado + JSON de salida, y tarifas de referencia (Artificial Analysis / fallback: gpt-5.6-sol ≈ $2/$8 por 1M input/output; deepseek-v4-pro ≈ $0.55/$2.19). **No incluye tokens de reasoning interno** (`reasoning: medium`), así que el coste real de OpenAI suele ser mayor.

---

## Tabla comparativa

| Métrica | `main_prompt` OpenAI | `main_prompt` DeepSeek | `main_prompt_enhanced_00` OpenAI | `main_prompt_enhanced_00` DeepSeek | `main_prompt_enhanced_01` OpenAI | `main_prompt_enhanced_01` DeepSeek | `main_prompt_enhanced_02` OpenAI | `main_prompt_enhanced_02` DeepSeek |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Coste aprox. (USD) | ~$0.18 | ~$0.05 | ~$0.18 | ~$0.05 | ~$0.20 | ~$0.05 | ~$0.20 | ~$0.05 |
| Tokens (in/out) | 71k / 4.2k | 71k / 4.0k | 73k / 4.2k | 73k / 6.0k | 74k / 6.9k | 74k / 5.5k | 74k / 6.6k | 74k / 5.8k |
| Calidad global | 6 | 6 | 5 | 7 | 8 | 6.5 | **8.5** | 7 |
| Alineación prompt | 5 | 5 | 6 | 6 | 8 | 7 | **9** | 8 |
| Client-based | 4 | 4 | 5 | 5 | 6 | 5 | **8** | 6 |
| Costes vía operadores | 3 | 3 | 7 | 5 | 9 | 8 | 9 | 8 |
| CVRP completo (carga/flota/via-depot) | 7 | 7 | 3 | 6 | 8 | 7 | **9** | 8 |
| Multi-componente | 2 | 2 | 3 | 3 | 8 | 4 | **9** | 7 |
| State-dependent | 8 | 8 | 7 | 8 | 7 | 8 | **9** | 9 |
| Robustez | 5 | 4 | 6 | 5 | 7 | 5 | **8** | 6 |
| Interpretabilidad | 6 | 6 | 7 | 6 | 9 | 8 | **9** | 8 |
| Riesgo compilación | 2 | 2 | 2 | 4 | 3 | 6 | 2 | **7** |

**Total aprox. (8 runs):** ~**$0.97** (solo tokens visibles; reasoning no contabilizado).

**Leyenda rápida de filas**

- **Coste aprox. (USD):** coste de API estimado por run (prompt + respuesta JSON).
- **Tokens (in/out):** tokens de entrada (prompt completo) y salida (JSON generado), redondeados.
- **Calidad global:** valoración integrada de la heurística como candidata útil para PlanForge.
- **Alineación prompt:** cuánto cumple lo que el prompt pedía (MUST, catálogo, checklist, JSON extendido).
- **Client-based:** adaptación a patrones de instancia sin memorizar literales; penaliza tours fijos ciegos.
- **Costes vía operadores:** uso de `metric_operator_cost_from_initial_values` sobre acciones grounded en lugar de parsear PNEs frágiles.
- **CVRP completo:** carga, capacidad, flota, visit-via-depot y retorno al depot.
- **Multi-componente:** descomposición explícita tipo `h_routing + h_capacity + h_fleet`.
- **State-dependent:** la estimación depende de ubicación, carga, vehículos y clientes no visitados actuales.
- **Robustez:** tolerancia a PNEs compilados/opacos, simetría, datos parciales.
- **Interpretabilidad:** calidad de `what`/`why`/`design`/`known_limitations`/`instance_patterns`.
- **Riesgo compilación:** probabilidad de fallo por imports, tipos o APIs incorrectas.

---

## Conclusiones transversales

1. **Los prompts enhanced mejoran de forma clara** respecto a `main_prompt` y `main_prompt_enhanced_00`: más operadores grounded, mejor documentación y diseño multi-componente.
2. **OpenAI responde mejor al checklist del prompt enhanced**, especialmente en `_02`: incluye `instance_patterns`, evita tour fijo y separa componentes con intención clara.
3. **DeepSeek tiende a producir greedy completos state-dependent**, pero repite errores de imports (`planforge_sas::state::ExplicitFact`, `metric_operator_cost_from_initial_values` en `evaluation` en vez de `heuristics::ff`).
4. **El parseo directo de PNEs `travel-cost(...)` es frágil** frente a instancias donde PlanForge compila expresiones derivadas; los enhanced que priorizan operadores grounded son más sólidos.
5. **`main_prompt_enhanced_00` OpenAI es coherente pero incompleto para CVRP**: el propio `why` justifica omitir capacidad/demanda; útil como baseline geométrico, no como heurística CVRP plena.
6. **Componentes de capacidad/flota a veces quedan muertos**: DeepSeek enhanced_01 deja `h_capacity = 0.0` siempre; enhanced_02 DeepSeek sí activa componentes pero con fórmulas ad hoc.
7. **Falta validación empírica**: ningún run se ha compilado ni benchmarked aún contra FF en PlanForge.

---

# main_prompt

Prompt base: pide heurística Rust para PlanForge CVRP con descubrimiento semántico de facts y variables numéricas, sin muchos requisitos duros ni catálogo de técnicas.

## OpenAI (gpt-5.6-sol)

**Idea central:** greedy capacitado nearest-neighbor con matriz de distancias reconstruida desde PNEs simples (`travel-cost`, `demand`, `capacity`, `max-vehicles`), relleno de arcos faltantes vía depot o coste medio, y penalización de dead-ends por recursos.

**Fortalezas**

- CVRP razonablemente completo: carga, capacidad, flota, via-depot, retorno final.
- State-dependent real: simula continuación desde ubicación y carga actuales.
- Código legible, imports correctos, estructura PlanForge estándar.
- Detección de infeasibilidad agregada (`remaining_demand` vs capacidad residual + flota).

**Debilidades**

- Costes extraídos solo de PNEs parseables, no de operadores grounded.
- Un solo término de routing (sin componentes separados de capacidad/flota).
- Relleno de matriz con heurísticas conservadoras puede distorsionar costes reales.
- `what`/`why` correctos pero sin metadatos extendidos (`design`, limitaciones).

**Calidad estimada:** 6/10 — buen primer intento funcional, pero frágil ante PNEs compilados.

## DeepSeek (deepseek-v4-pro)

**Idea central:** muy similar a OpenAI — NN greedy capacitado con listas `nearest_from` preordenadas, parseo semántico de facts y PNEs por sufijos (`ends_with("demand(")`).

**Fortalezas**

- Greedy state-dependent bien implementado con elección direct vs via-depot.
- Buffers reutilizables (`RefCell`) para evaluación repetida.
- Misma cobertura CVRP que OpenAI en intención.

**Debilidades**

- Parseo de PNEs aún más frágil (búsqueda de subcadenas).
- Simetrización manual de distancias y relleno con `1e9` poco principled.
- Sin operadores grounded ni componentes separados.
- Lógica de carga al salir del depot algo ad hoc (`load.unwrap_or_else(|| demands[cur])`).

**Calidad estimada:** 6/10 — comparable a OpenAI, algo más hacky en construcción de matriz.

---

# main_prompt_enhanced_00

Variante con prompt más largo/agotado de contexto. OpenAI interpreta la imposibilidad de APIs de demanda y se centra en geometría pura; DeepSeek intenta MST + greedy con recuperación heurística de capacidad.

## OpenAI (gpt-5.6-sol)

**Idea central:** reconstrucción de costes desde efectos de operadores `visit`/`return-to-depot`; en evaluación combina **nearest-neighbor completion** + **MST** sobre clientes no visitados. Explícitamente **no modela capacidad ni demanda**.

**Fortalezas**

- Buen uso de operadores para costes estáticos (más robusto que parsear PNEs).
- Dual estimador MST + NN reduce miopía del greedy puro.
- `why` muy honesto sobre limitaciones y generalización.
- Buffers preasignados, código relativamente limpio.

**Debilidades**

- **No es heurística CVRP completa**: ignora carga, capacidad y límite de flota.
- Sin via-depot en la simulación de completado.
- Para el dominio CVRP real, puede crear plateaus donde estados con misma geometría pero distinta factibilidad reciben el mismo h.

**Calidad estimada:** 5/10 — buena heurística geométrica, insuficiente para CVRP capacitado.

## DeepSeek (deepseek-v4-pro)

**Idea central:** construcción pesada en `new()` — escaneo de operadores, efectos numéricos, matrices direct/via-depot/return; en evaluación devuelve el **máximo** entre estimador MST y greedy NN capacitado cuando hay datos de carga.

**Fortalezas**

- Intenta cubrir CVRP completo con hacks de descubrimiento de capacidad/flota.
- MST + greedy es diseño razonable para conectividad restante.
- Greedy capacitado cuando `load/capacity` están disponibles.

**Debilidades**

- Detección de `capacity`/`max_vehicles` por heurística (“variables estáticas no referenciadas”) es frágil.
- Código largo y difícil de auditar (~22k chars).
- Sin metadatos extendidos ni `instance_patterns`.
- Mayor riesgo de edge cases en instancias con PNEs no estándar.

**Calidad estimada:** 7/10 — más ambicioso que OpenAI en CVRP, pero más frágil en construcción.

---

# main_prompt_enhanced_01

Primer prompt enhanced: requisitos duros, catálogo de técnicas, JSON extendido (`design`, `known_limitations`), multi-componente obligatorio, reconstrucción de costes vía operadores.

## OpenAI (gpt-5.6-sol)

**Idea central:** precomputa costes directed (visit, via-depot, return) desde operadores; construye un **`tour_order` fijo** (nearest-neighbour sobre clientes); en evaluación simula completado sobre ese ciclo con transiciones direct/via-depot según carga y flota; suma `h_routing + h_capacity + h_fleet_pressure`.

**Fortalezas**

- Mejor alineación con prompt enhanced: operadores, 3 componentes, metadatos ricos.
- Detección de dead-ends con penalización finita grande.
- `design` y `known_limitations` explícitos y honestos (admite no-admisibilidad y tour fijo).
- Imports correctos (`planforge_search::heuristics::ff::metric_operator_cost_from_initial_values`).

**Debilidades**

- **`tour_order` fijo**: no reordena según prefijo de visitas; penaliza client-based.
- Evaluación O(n) sobre ciclo fijo, pero puede subestimar/sobreestimar tras prefijos atípicos.
- Componentes de capacidad/flota pueden double-count según admite el propio `known_limitations`.

**Calidad estimada:** 8/10 — mejor run enhanced inicial; principal defecto es el tour precomputado ciego.

## DeepSeek (deepseek-v4-pro)

**Idea central:** similar en espíritu — operadores grounded, listas NN precomputadas, greedy capacitado state-dependent, penalización por infeasibilidad de recursos.

**Fortalezas**

- Greedy **re-anclado en cada estado** (mejor que tour fijo de OpenAI en este aspecto).
- Buen `what`/`why`/`design`/`known_limitations`.
- Usa `metric_operator_cost_from_initial_values` sobre operadores parseados.

**Debilidades**

- **`h_capacity = 0.0` siempre** — el componente de capacidad prometido no aporta nada.
- Import sospechoso `planforge_sas::task::Fact` en lugar del patrón `ExplicitFact` estándar.
- `metric_operator_cost_from_initial_values` importado desde `evaluation` (ruta probablemente incorrecta).
- Solo 2 componentes efectivos (routing + dead-end), no 3 como pide el prompt.

**Calidad estimada:** 6.5/10 — buena intuición algorítmica, ejecución incompleta del diseño multi-componente.

---

# main_prompt_enhanced_02

Copia de `_01` + **Client Specialization Requirement**: prohíbe tour fijo ciego, exige campo `instance_patterns`, checklist ampliado.

## OpenAI (gpt-5.6-sol)

**Idea central:** precomputa arcos directed y mínimos entrantes; en evaluación elige la **primera transición legal** desde el estado actual (direct o via-depot), suma continuación geométrica relajada (`min_incoming` sobre no visitados) + retorno al depot; añade `h_capacity` (recargas obligatorias) y `h_fleet` (presión de flota). Incluye `instance_patterns` explícitos.

**Fortalezas**

- **Mejor candidata global**: O(n) por estado, sin tour fijo, 3 componentes reales.
- Costes desde operadores grounded con fallback PNE mínimo.
- `instance_patterns`, `design` y `known_limitations` de alta calidad.
- State-dependent fuerte: primera transición + continuación anclada al prefijo real.
- Imports y APIs PlanForge correctos.

**Debilidades**

- Continuación por suma de `min_incoming` no forma un tour global consistente (admite el propio JSON).
- Componentes no admisibles; posible double-count de presión de depot.
- Test de factibilidad agregada no detecta todos los dead-ends de bin-packing.

**Calidad estimada:** 8.5/10 — referencia para siguiente fase (compilar + benchmark).

## DeepSeek (deepseek-v4-pro)

**Idea central:** precomputa matrices direct/via-depot, listas NN; greedy capacitado completo desde estado actual; añade componentes `h_capacity_component` y `h_vehicle_component` con fórmulas ad hoc; incluye `instance_patterns`.

**Fortalezas**

- Greedy state-dependent completo (direct vs via-depot en cada paso).
- Operadores como fuente primaria de costes; fallback PNE secundario.
- Metadatos extendidos presentes (`instance_patterns`, `design`, `known_limitations`).
- Penalización de dead-ends por demanda residual vs flota.

**Debilidades**

- **Imports incorrectos**: `planforge_sas::state::ExplicitFact` y `metric_operator_cost_from_initial_values` desde `evaluation`.
- Componentes de capacidad/flota con escalado arbitrario (`* 2.0`), no alineados con unidades de coste del dominio.
- Código más difícil de validar que la versión OpenAI del mismo prompt.
- **Alto riesgo de no compilar** sin corrección manual.

**Calidad estimada:** 7/10 — algoritmo sólido en papel, ejecución PlanForge dudosa por detalles de API.

---

## Ranking final (cualitativo, sin benchmark)

| Posición | Experimento | Motivo principal |
|---:|---|---|
| 1 | `main_prompt_enhanced_02` · OpenAI | O(n), sin tour fijo, 3 componentes, operadores, metadatos completos |
| 2 | `main_prompt_enhanced_01` · OpenAI | Muy buen diseño salvo `tour_order` fijo |
| 3 | `main_prompt_enhanced_02` · DeepSeek | Greedy completo + patterns; imports probablemente rotos |
| 4 | `main_prompt_enhanced_00` · DeepSeek | MST + greedy capacitado; construcción frágil |
| 5 | `main_prompt_enhanced_01` · DeepSeek | Operadores OK; `h_capacity = 0` |
| 6 | `main_prompt` · OpenAI | Baseline CVRP razonable; PNE-only |
| 7 | `main_prompt` · DeepSeek | Similar a #6, más hacks |
| 8 | `main_prompt_enhanced_00` · OpenAI | Geométrico sin capacidad/flota |

---

## Próximos pasos sugeridos

1. **Cerrar el protocolo train/val/test** (sección siguiente) y generar **hA** (familia A) y **hB** (familia P).
2. **Compilar y benchmark** en PlanForge: sanity en train, tuning en val, métricas finales solo en test; siempre vs FF.
3. **Corregir DeepSeek enhanced_02** (imports) y re-ejecutar, o usar post-procesado automático de imports en el pipeline.
4. **Cablear `hllm_schema()`** en `generate.py` y ampliar schema con campos enhanced si se quieren forzar por API.

---

## Protocolo train / val / test (familias A y P)

Objetivo: medir si una heurística **especializa** a su familia de entrenamiento o **generaliza** al otro dominio cliente. Mismo dominio PDDL (`domain.pddl`); dos distribuciones de instancias CVRPLib: **Augerat A** (`data/cvrplib/A/`, 27 instancias) y **Augerat P** (`data/cvrplib/P/`, 24 instancias).

### Roles de cada split

| Split | Qué es | Para qué sirve | Regla |
|---|---|---|---|
| **Train** | Instancias embebidas en el prompt | Generar hA / hB; el LLM las “estudia” | Puede usarse en análisis cualitativo; benchmark aquí = *sanity check* (compila, corre, señal vs FF) |
| **Val** | Instancias **no** vistas en el prompt | Elegir prompt/modelo/candidato antes del informe final | Permitido iterar; **no** reportar como resultado principal del TFM |
| **Test** | Instancias **no** vistas en prompt ni usadas para decidir | Métricas finales del experimento | **Congelar** antes de mirar resultados; una sola pasada |

`first_approach` usó solo **train A** (`A-n32-k5`, `A-n37-k5`). Aún no hay val/test ni hB.

### Split propuesto (congelar antes de mirar test)

**Familia A (`dataA`)**

| Split | Instancias | Motivo |
|---|---|---|
| Train | `A-n32-k5`, `A-n37-k5` | Ya usadas en `first_approach`; escala pequeña (~32–37 clientes, k=5) |
| Val | `A-n45-k6`, `A-n60-k9` | Tamaño/flota distintos; no estaban en el prompt |
| Test | `A-n39-k5`, `A-n69-k9`, `A-n80-k10` | Pequeña / mediana / grande (32→80 clientes) |

**Familia P (`dataB`)**

| Split | Instancias | Motivo |
|---|---|---|
| Train | `P-n22-k8`, `P-n40-k5` | Análogo a train A: pequeña + mediana en la familia P |
| Val | `P-n50-k10`, `P-n55-k7` | Escalas intermedias, distintas de train |
| Test | `P-n23-k8`, `P-n76-k5`, `P-n101-k4` | Pequeña / grande / máxima de la familia |

Todas las demás instancias de A y P quedan **fuera del protocolo inicial** (reserva para extensiones o repetición).

### Heurísticas a generar

| ID | Prompt train | Salida |
|---|---|---|
| **hA** | `domain.pddl` + **train dataA** (2 instancias A) | Código Rust de `enhanced_02` (u otro prompt elegido en val) |
| **hB** | `domain.pddl` + **train dataB** (2 instancias P) | Mismo prompt template, distintas instancias representativas |

Mismo prompt template para hA y hB (p. ej. `main_prompt_enhanced_02.md`); solo cambian las instancias representativas. Así el contraste es **familia de datos**, no variante de prompt.

### Matriz de evaluación (test únicamente)

Medir en PlanForge: **nodos expandidos**, **tiempo**, **¿solución?** — siempre vs **FF** en la misma instancia.

|  | **dataA test** | **dataB test** |
|---|---|---|
| **hA** | in-distribution (especialización esperada) | transferencia A→P |
| **hB** | transferencia P→A | in-distribution (especialización esperada) |
| **FF** | baseline dominio | baseline dominio |

Interpretación esperada:

- **Diagonal fuerte** (hA en A, hB en P) → el pipeline especializa bien por familia cliente.
- **Diagonal ≫ off-diagonal** → patrones de A y P no son intercambiables; client-based funciona.
- **Off-diagonal ≈ diagonal** → heurística genérica CVRP (poco especialización).
- **Peor que FF en alguna celda** → la heurística no ayuda (o no compila) en ese régimen.

### Orden de trabajo recomendado

1. **Sanity (train):** compilar candidatos de `first_approach` (`enhanced_02` OpenAI, `enhanced_01` OpenAI) en `A-n32-k5`, `A-n37-k5`.
2. **Val A:** elegir candidato hA definitivo en `A-n45-k6`, `A-n60-k9`.
3. **Generar hB** con train P (`P-n22-k8`, `P-n40-k5`); val en `P-n50-k10`, `P-n55-k7`.
4. **Test congelado:** ejecutar matriz 2×2 + FF en las 6 instancias test (3+3).
5. **Opcional:** repetir con otro prompt (`enhanced_01`) solo si val lo justifica — sin tocar test.

### Métricas a registrar (plantilla)

Por celda `(h, instancia)`:

- compila (sí/no)
- nodos expandidos (h vs FF)
- ratio `expanded_h / expanded_FF`
- tiempo de búsqueda
- coste de plan encontrado (sanity de factibilidad)

---

## Archivos de referencia

```
results/first_approach/
├── main_prompt/
│   ├── out - openai - gpt-5.6-sol - medium.rs
│   ├── out - openai - gpt-5.6-sol - medium.json
│   ├── out - deepseek - deepseek-v4-pro - medium.rs
│   └── out - deepseek - deepseek-v4-pro - medium.json
├── main_prompt_enhanced_00/
│   ├── out - openai - gpt-5.6-sol - medium.rs
│   ├── out - openai - gpt-5.6-sol - medium.json
│   ├── out - deepseek - deepseek-v4-pro - medium.rs
│   └── out - deepseek - deepseek-v4-pro - medium.json
├── main_prompt_enhanced_01/
│   ├── out - openai - gpt-5.6-sol - medium.rs
│   ├── out - openai - gpt-5.6-sol - medium.json
│   ├── out - deepseek - deepseek-v4-pro - medium.rs
│   └── out - deepseek - deepseek-v4-pro - medium.json
├── main_prompt_enhanced_02/
│   ├── out - openai - gpt-5.6-sol - medium.rs
│   ├── out - openai - gpt-5.6-sol - medium.json
│   ├── out - deepseek - deepseek-v4-pro - medium.rs
│   └── out - deepseek - deepseek-v4-pro - medium.json
└── INSIGHTS.md
```

Cada `out - … .rs` es el código Rust; el `.json` del mismo nombre guarda solo los campos de explicabilidad (`what`, `why`, y metadatos extra si los hay).
