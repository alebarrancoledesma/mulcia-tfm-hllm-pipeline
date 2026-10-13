# INSIGHTS — First Approach (HLLM Heuristic Generation)

Análisis cualitativo de las 8 ejecuciones del pipeline sobre instancias representativas CVRPLib (`A-n32-k5`, `A-n37-k5`), dominio PlanForge CVRP, modelos **gpt-5.6-sol** (OpenAI) y **deepseek-v4-pro** (DeepSeek), reasoning `medium`.

**Nota:** estas valoraciones se basan en revisión estática del JSON generado (`code`, `what`, `why`, metadatos) y, solo para `main_prompt`, en una pasada de compilación/ejecución en PlanForge. Los otros seis `.rs` no se han compilado aún. No hay métricas de nodos expandidos ni de calidad de búsqueda.

**Escala usada en la tabla:** 1–10 (mayor = mejor), excepto *Riesgo compilación* (mayor = peor).

**Coste aprox.:** estimado con `tiktoken` (`o200k_base`) sobre el prompt materializado + JSON reconstruido (`code` + campos de explicabilidad), y tarifas de referencia (Artificial Analysis / fallback: gpt-5.6-sol ≈ $2/$8 por 1M input/output; deepseek-v4-pro ≈ $0.55/$2.19). **No incluye tokens de reasoning interno** (`reasoning: medium`), así que el coste real de OpenAI suele ser mayor.

---

## Tabla comparativa

| Métrica | `main_prompt` OpenAI | `main_prompt` DeepSeek | `main_prompt_enhanced_00` OpenAI | `main_prompt_enhanced_00` DeepSeek | `main_prompt_enhanced_01` OpenAI | `main_prompt_enhanced_01` DeepSeek | `main_prompt_enhanced_02` OpenAI | `main_prompt_enhanced_02` DeepSeek |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Coste aprox. (USD) | ~$0.18 | ~$0.05 | ~$0.18 | ~$0.05 | ~$0.20 | ~$0.05 | ~$0.21 | ~$0.05 |
| Tokens (in/out) | 72k / 4.0k | 72k / 4.9k | 73k / 4.3k | 73k / 5.6k | 74k / 6.6k | 74k / 6.0k | 75k / 7.0k | 75k / 6.3k |
| Calidad global | 7 | 5 | 6 | 7 | 7 | **7.5** | **7.5** | **8** |
| Alineación prompt | 6 | 6 | 6 | 6 | 8 | 8 | **9** | 8 |
| Client-based | 5 | 5 | 5 | 5 | 6 | 6 | **7** | **7** |
| Costes vía operadores | 8 | **9** | 7 | 8 | **9** | 8 | **9** | 8 |
| CVRP completo (carga/flota/via-depot) | 4 | 7 | 4 | 7 | 8 | 8 | **9** | 8 |
| Multi-componente | 3 | 2 | 3 | 4 | 8 | 7 | 8 | 8 |
| State-dependent | 8 | 8 | 7 | 8 | 7 | 8 | 8 | **9** |
| Robustez | **8** | 4 | 7 | 6 | 4 | 6 | 3 | 7 |
| Interpretabilidad | 6 | 6 | 7 | 6 | **9** | 8 | **9** | 8 |
| Riesgo compilación | **1** | **8** | 2 | 3 | 3 | 3 | 3 | 3 |

**Total aprox. (8 runs):** ~**$0.97** (solo tokens visibles; reasoning no contabilizado).

**Leyenda rápida de filas**

- **Coste aprox. (USD):** coste de API estimado por run (prompt + respuesta JSON).
- **Tokens (in/out):** tokens de entrada (prompt completo) y salida (JSON reconstruido), redondeados.
- **Calidad global:** valoración integrada de la heurística como candidata útil para PlanForge.
- **Alineación prompt:** cuánto cumple lo que el prompt pedía (MUST, catálogo, checklist, JSON extendido).
- **Client-based:** adaptación a patrones de instancia sin memorizar literales; penaliza tours fijos ciegos.
- **Costes vía operadores:** uso de `metric_operator_cost_from_initial_values` sobre acciones grounded en lugar de parsear PNEs frágiles.
- **CVRP completo:** carga, capacidad, flota, visit-via-depot y retorno al depot.
- **Multi-componente:** descomposición explícita tipo `h_routing + h_capacity + h_fleet`.
- **State-dependent:** la estimación depende de ubicación, carga, vehículos y clientes no visitados actuales.
- **Robustez:** tolerancia a PNEs compilados/opacos, simetría, datos parciales; penaliza `new()` que aborta si falta `capacity()`.
- **Interpretabilidad:** calidad de `what`/`why`/`design`/`known_limitations`/`instance_patterns`.
- **Riesgo compilación:** probabilidad de fallo por imports, tipos o APIs incorrectas. 1 = compiló de hecho; 8 = falló de hecho.

---

## Conclusiones transversales

1. **El bloque PlanForge API del prompt corrigió el fallo masivo de imports** de `first_approach_old`: las 8 salidas importan `ExplicitFact` y `metric_operator_cost_from_initial_values` desde `planforge_sas::numeric_task`. El riesgo ya no es “API inventada”, sino tipos Rust y `new()` demasiado estrictos.
2. **Hay evidencia empírica solo en `main_prompt`:** OpenAI **compila y corre** (timeout 30s en `A-n33-k5`, sin error de construcción). DeepSeek **no compila** (`(*a).cmp(*b)`: `expected &usize, found usize`).
3. **OpenAI `main_prompt` y `enhanced_00` eligen geometría y evitan recursos opacos.** Funciona (al menos el primero), pero no es una heurística CVRP plena: ignoran carga, capacidad y flota; OpenAI `main_prompt` ni siquiera usa `visit-via-depot`.
4. **OpenAI `enhanced_01`/`_02` hacen lo contrario:** diseño CVRP rico y metadatos excelentes, pero `new()` exige `load()`, `vehicles()`, `capacity()`, `max_vehicles` y demandas recuperables. En PlanForge esas constantes suelen ser `derived!...`; el patrón ya mató una generación anterior en construcción.
5. **DeepSeek degrada mejor cuando faltan PNEs:** `main_prompt` y `enhanced_02` tratan capacidad/demanda como opcionales (`Option` / `0.0` / `f64::MAX`). El precio es un fallo de tipos en `main_prompt` y código más largo.
6. **Todos reconstruyen costes desde operadores grounded.** El parseo directo de `travel-cost(...)` ya no es la fuente primaria; el menú de API del prompt se nota.
7. **Los prompts enhanced siguen ganando en documentación:** `_01` aporta `design`/`fallbacks`/`known_limitations`; `_02` añade `instance_patterns`. `main_prompt` y `_00` se quedan en `what`/`why`.
8. **Falta el benchmark real** de `_00`/`_01`/`_02` (compilación + nodos + tiempo vs FF). Hasta entonces, `_02` DeepSeek es la mejor candidata sobre el papel; `main_prompt` OpenAI es la única demostrablemente ejecutable.

---

# main_prompt

Prompt base: pide heurística Rust para PlanForge CVRP con descubrimiento semántico de facts y variables numéricas, sin muchos requisitos duros ni catálogo de técnicas. El bloque API documenta imports reales, `EvaluationState` y que `capacity`/`demand`/`travel-cost` suelen ser derivados.

## OpenAI (gpt-5.6-sol)

**Idea central:** reconstruye la matriz directed desde operadores `visit` y `return-to-depot`; en cada estado combina dos completions nearest-neighbor, un MST y una relajación de grado sobre depot + ubicación actual + no visitados. Devuelve un blend inflado ×1.15. **No modela carga, capacidad ni flota.**

**Fortalezas**

- **Compiló y corrió** en PlanForge; el timeout en `A-n33-k5` es de búsqueda, no de construcción.
- Costes desde operadores grounded; no depende de PNEs `travel-cost` compilados.
- State-dependent real (ubicación + conjunto no visitado) con dual NN + MST/grado.
- `new()` no aborta por recursos ausentes: máxima robustez del lote OpenAI.

**Debilidades**

- **No es heurística CVRP completa:** ignora carga, capacidad, flota y `visit-via-depot`.
- Un solo término de routing (el blend no separa `h_capacity` / `h_fleet`).
- La inflación ×1.15 es un parche admitido para recargas que no puede reconstruir.
- `what`/`why` correctos pero sin metadatos extendidos.

**Calidad estimada:** 7/10 — única candidata demostrablemente ejecutable; baseline geométrico sólido, insuficiente como CVRP capacitado.

## DeepSeek (deepseek-v4-pro)

**Idea central:** precomputa matrices `visit` / `visit-via-depot` / `return-to-depot` y listas NN; greedy state-dependent que elige la transición legal más barata (directa vs via-depot). Si encuentra `load`/`capacity`/`vehicles` las usa; si no, **degrada** a routing puro.

**Fortalezas**

- CVRP más completo que OpenAI en el mismo prompt: via-depot, retorno y recursos opcionales.
- Costes 100% vía operadores grounded; descubrimiento semántico por sufijos.
- El `why` justifica bien el degrade graceful que el bloque API pedía.

**Debilidades**

- **No compila:** `(*a).cmp(*b)` en `sorted_customer_indices` (el closure de `sort_by` ya tiene `&usize`).
- Sin componentes separados; un solo greedy.
- `what`/`why` solo; sin `design` ni limitaciones.
- Hasta corregir el `.cmp`, no es candidata usable.

**Calidad estimada:** 5/10 — mejor diseño CVRP que OpenAI en este prompt, pero inútil mientras no compile.

---

# main_prompt_enhanced_00

Variante con prompt más largo/agotado de contexto. OpenAI se centra en geometría (NN + MST) e ignora capacidad; DeepSeek promedia MST y un greedy capacitado con recuperación heurística de recursos.

## OpenAI (gpt-5.6-sol)

**Idea central:** clientes y depot desde nombres de operadores; costes de `visit` y `return-to-depot`; en evaluación `0.65 · NN + 0.595 · MST`. El NN permite transición directa o **via-depot compuesto** (`from→depot + depot→to`), no el coste del operador `visit-via-depot`. Explícitamente **no modela capacidad ni demanda**.

**Fortalezas**

- Imports y estilo muy cercanos al `main_prompt` OpenAI (bajo riesgo de compilación).
- Dual NN + MST, buffers reutilizables, `why` honesto sobre no-admisibilidad.
- Via-depot aparece al menos como modo de movimiento, a diferencia de `main_prompt` OpenAI.

**Debilidades**

- **Sigue sin ser CVRP capacitado:** misma geometría, mismos plateaus de factibilidad.
- Parsea `visit-via-depot` solo para descubrir clientes; **no guarda su coste de operador**.
- Sin metadatos enhanced (`design`, `instance_patterns`).

**Calidad estimada:** 6/10 — heurística geométrica limpia, un paso por encima de `main_prompt` OpenAI en via-depot, igual de incompleta para CVRP.

## DeepSeek (deepseek-v4-pro)

**Idea central:** construcción pesada en `new()` — operadores, facts `loc`/`visited`, `load`/`vehicles`, `capacity`/`max_vehicles` por substring, demandas desde efectos de asignación o PNE `demand`. Evaluación: **promedio** de MST (con aristas de start/depot) y greedy capacitado direct/via-depot.

**Fortalezas**

- Intenta CVRP completo sin abortar si falta capacidad (`INF` por defecto).
- MST + greedy es un diseño razonable para conectividad restante vs. ruta realista.
- Demandas extraídas de efectos de `load` cuando el PNE `demand(...)` no existe: fallback más inteligente que el de OpenAI `_01`/`_02`.

**Debilidades**

- Detección de `capacity`/`load` por `name.contains(...)` es frágil (puede ligar un derivado opaco o el fluent equivocado).
- Código largo, más difícil de auditar; sin metadatos extendidos.
- El promedio 50/50 no está justificado en unidades de coste del dominio.

**Calidad estimada:** 7/10 — más ambicioso y usable en construcción que OpenAI `_00`; sigue siendo un prototipo de `_01`/`_02`.

---

# main_prompt_enhanced_01

Primer prompt enhanced: requisitos duros, catálogo de técnicas, JSON extendido (`design`, `known_limitations`), multi-componente obligatorio, reconstrucción de costes vía operadores.

## OpenAI (gpt-5.6-sol)

**Idea central:** precomputa costes directed (`visit`, `visit-via-depot`, `return`) y nearest-links; en evaluación lee load/vehicles, testa factibilidad agregada de flota y devuelve `h_routing + h_capacity`. `h_routing` = primera transición legal (direct o via-depot) + nearest-link escalado + retorno. `h_capacity` cobra recargas obligatorias × (min retorno + min salida de depot). **`new()` falla** si no recupera `load()`, `vehicles()`, `capacity()`, `max_vehicles` o demandas.

**Fortalezas**

- Mejor alineación enhanced: operadores, dos componentes reales, metadatos ricos y honestos.
- **No usa tour fijo** (a diferencia de la generación `first_approach_old` del mismo prompt).
- Dead-ends con penalización finita; O(n) por estado.
- `known_limitations` admite nearest-link laxo y que el bin-packing no está resuelto.

**Debilidades**

- **`new()` frágil:** el propio `fallbacks` dice que aborta si faltan depot legs, recursos o demandas. En este dominio `capacity`/`demand` suelen ser `derived!...`.
- Continuación por suma de nearest-links no forma un tour consistente.
- Solo 2 componentes (`h_routing + h_capacity`); el prompt pedía también presión de flota explícita.
- Sin evidencia de compilación aún; el riesgo real es construcción, no imports.

**Calidad estimada:** 7/10 — diseño enhanced limpio; el hard-fail de recursos replica el error de la generación anterior.

## DeepSeek (deepseek-v4-pro)

**Idea central:** operadores grounded + listas NN y via-depot preordenadas; greedy capacitado state-dependent (`h_routing`) + `h_resource` (vehículos extra × 2 × arco medio a depot). Exige `load()` y `vehicles()`; `capacity` cae a `1e9` si falta; demandas ausentes → 0.

**Fortalezas**

- Greedy **re-anclado en cada estado**, con elección real direct vs via-depot.
- `h_resource` no es un stub a 0 (a diferencia de DeepSeek `_01` en `first_approach_old`).
- JSON extendido completo y honesto (greedy puede gastar más vehículos; fallback 2×depot).
- Fallback de arcos via-depot = suma de piernas depot si falta el operador.

**Debilidades**

- Sigue exigiendo `load()`/`vehicles()` en `new()`; menos duro que OpenAI, pero no degrade total.
- Demandas a 0 debilitan el test de recursos (lo admite `known_limitations`).
- Escalado `* 2.0 * avg_depot_arc` es ad hoc.
- Greedy worst-case O(n²) por estado.

**Calidad estimada:** 7.5/10 — mejor equilibrio enhanced_01 entre diseño CVRP y tolerancia a PNEs opacos.

---

# main_prompt_enhanced_02

Copia de `_01` + **Client Specialization Requirement**: prohíbe tour fijo ciego, exige campo `instance_patterns`, checklist ampliado.

## OpenAI (gpt-5.6-sol)

**Idea central:** construye un **ciclo geográfico NN** en `new()`; en cada estado lo **re-ancla** al no visitado más cercano, simula completion capacitada (direct vs via-depot + flota) y suma `h_routing + h_capacity` (overhead de ruta × recargas). Incluye `instance_patterns`. **`new()` falla** si no recupera routing usable o `load`/`vehicles`/`capacity`/`max_vehicles`/demandas.

**Fortalezas**

- Máxima alineación `_02`: patterns, design, fallbacks, known_limitations, O(n) anunciado.
- El ciclo no es ciego: se re-ancla y se filtra por el prefijo real (cumple el espíritu anti-tour-fijo mejor que `_01` old).
- Tres ingredientes reales: routing capacitado, recargas, dead-end de flota agregada.
- Costes de operador con fallback PNE y composición por piernas depot; no asume simetría.

**Debilidades**

- **El `new()` más estricto del lote:** un `capacity` derivado o una demanda irrecuperable tumba la heurística antes de buscar.
- El ciclo NN puede empaquetar mal la demanda (lo admite el JSON) y penalizar estados factibles con otro orden.
- Sin evidencia de compilación; el riesgo dominante es construcción, no tipos.

**Calidad estimada:** 7.5/10 — mejor diseño OpenAI del lote; la candidatura real depende de que PlanForge exponga esos fluents.

## DeepSeek (deepseek-v4-pro)

**Idea central:** greedy capacitado anclado al estado (nearest feasible direct, si no via-depot si queda flota) + `h_capacity_penalty` + `h_vehicle_pressure`. Demanda ausente → 0; capacidad ausente → `f64::MAX`. Incluye `instance_patterns`.

**Fortalezas**

- **Mejor candidata global sobre el papel:** 3 componentes, via-depot real, degrade graceful, metadatos `_02` completos.
- Imports correctos; no replica el `.cmp` roto de `main_prompt` DeepSeek.
- Dead-ends por demanda residual vs flota; routing sigue vivo aunque falten recursos.
- Client-based explícito en `instance_patterns` (flota pequeña, demandas heterogéneas, matriz densa).

**Debilidades**

- Greedy local sin lookahead; el JSON lo admite.
- Componentes de presión con umbrales/`LARGE_PENALTY` ad hoc.
- Evaluación worst-case O(k²) sobre no visitados (aceptable en A-n32/A-n37, peor en A grandes).
- Aún **sin compilar** en esta carpeta; el riesgo residual es de tipos/edge cases, no de imports.

**Calidad estimada:** 8/10 — referencia cualitativa para la siguiente fase de compile + benchmark.

---

## Ranking final (cualitativo; compile/run solo en `main_prompt`)

| Posición | Experimento | Motivo principal |
|---:|---|---|
| 1 | `main_prompt_enhanced_02` · DeepSeek | Greedy capacitado + 3 componentes + degrade graceful + `instance_patterns` |
| 2 | `main_prompt_enhanced_01` · DeepSeek | Greedy re-anclado + `h_resource` real; exige `load`/`vehicles` |
| 3 | `main_prompt` · OpenAI | **Única que compiló y corrió**; geométrica, sin capacidad |
| 4 | `main_prompt_enhanced_02` · OpenAI | Mejor diseño OpenAI; `new()` aborta si faltan recursos |
| 5 | `main_prompt_enhanced_01` · OpenAI | First-transition + nearest-link; mismo hard-fail de `capacity()` |
| 6 | `main_prompt_enhanced_00` · DeepSeek | MST + greedy capacitado; construcción heurística de recursos |
| 7 | `main_prompt_enhanced_00` · OpenAI | NN + MST limpio; via-depot compuesto, sin capacidad |
| 8 | `main_prompt` · DeepSeek | Buen diseño CVRP; **no compila** (`.cmp`) |

---

## Archivos de referencia

```
experiments/first_approach/
├── main_prompt/
│   ├── prompt - openai - gpt-5.6-sol - medium.md
│   ├── prompt - deepseek - deepseek-v4-pro - medium.md
│   ├── out - openai - gpt-5.6-sol - medium.rs
│   ├── out - openai - gpt-5.6-sol - medium.json
│   ├── out - deepseek - deepseek-v4-pro - medium.rs
│   └── out - deepseek - deepseek-v4-pro - medium.json
├── main_prompt_enhanced_00/
│   ├── prompt - openai - gpt-5.6-sol - medium.md
│   ├── prompt - deepseek - deepseek-v4-pro - medium.md
│   ├── out - openai - gpt-5.6-sol - medium.rs
│   ├── out - openai - gpt-5.6-sol - medium.json
│   ├── out - deepseek - deepseek-v4-pro - medium.rs
│   └── out - deepseek - deepseek-v4-pro - medium.json
├── main_prompt_enhanced_01/
│   ├── prompt - openai - gpt-5.6-sol - medium.md
│   ├── prompt - deepseek - deepseek-v4-pro - medium.md
│   ├── out - openai - gpt-5.6-sol - medium.rs
│   ├── out - openai - gpt-5.6-sol - medium.json
│   ├── out - deepseek - deepseek-v4-pro - medium.rs
│   └── out - deepseek - deepseek-v4-pro - medium.json
├── main_prompt_enhanced_02/
│   ├── prompt - openai - gpt-5.6-sol - medium.md
│   ├── prompt - deepseek - deepseek-v4-pro - medium.md
│   ├── out - openai - gpt-5.6-sol - medium.rs
│   ├── out - openai - gpt-5.6-sol - medium.json
│   ├── out - deepseek - deepseek-v4-pro - medium.rs
│   └── out - deepseek - deepseek-v4-pro - medium.json
└── INSIGHTS.md
```

Cada `out - … .rs` es el código Rust; el `.json` del mismo nombre guarda solo los campos de explicabilidad (`what`, `why`, y metadatos extra si los hay). El `prompt - … .md` es el prompt materializado enviado al modelo.
