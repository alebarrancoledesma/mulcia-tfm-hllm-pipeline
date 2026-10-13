# first_approach — memoria de prompts y tandas

Para retomar el hilo: qué pedían los prompts, qué cambió y por qué, y qué se vio en cada carpeta de [`= old/`](= old/).

## Constantes del experimento

- **Dominio:** CVRP PlanForge (`data/processed/pddl/domain.pddl`). Fluents dinámicos: `load`, `vehicles`. Límites estáticos: `capacity`, `max_vehicles`. Costes: `travel-cost`. Acciones: `visit`, `visit-via-depot`, `return-to-depot`.
- **Modelos:** OpenAI `gpt-5.6-sol` y DeepSeek `deepseek-v4-pro`, reasoning `medium`. 4 templates × 2 modelos = 8 candidatos por tanda.
- **Templates** (en `prompts/`, de menos a más exigentes):
  - `main_prompt.md` — diseño libre con API PlanForge.
  - `main_prompt_enhanced_00.md` — más guía de construcción vs evaluación.
  - `main_prompt_enhanced_01.md` — MUST/MUST NOT + catálogo de técnicas + JSON largo (`design`, `fallbacks`, …).
  - `main_prompt_enhanced_02.md` — lo de `_01` más especialización a la familia cliente (`instance_patterns`, no tour fijo calculado solo en `new()`).
- **Búsqueda cuando hay métricas:** GBFS 300s, mismas instancias que el prompt, más `blind` / `ff` / `lmcutnumeric`. CSV `metrics.csv`, logs en `logs/`.
- **Por qué P y no A:** A/B son grandes (n≥31) y P reutiliza A/B/E en las instancias grandes. El split oficial de P pequeño es `P-n16-k8`, `P-n20-k2`, `P-n23-k8` (prompt). Ver `data/README.md`.

Las tandas cerradas están archivadas. Nombres vivos que se usaron al generar: `first_approach` / `first_approach_01` → old5; `first_approach_02` → old6. No hay `first_approach_old2`.

| Carpeta | Instancias en el prompt | Texto de fluents / API | Qué se sacó |
|---|---|---|---|
| [`= old/first_approach_old`](= old/first_approach_old/) | A-n32-k5, A-n37-k5 | `get_fact_name`; sin bloque corto de PNEs | Sobre todo cualitativo (`INSIGHTS.md`) |
| [`= old/first_approach_old3`](= old/first_approach_old3/) | A-n32-k5, A-n37-k5 | Aparece *Observed CVRP names* | Cualitativo + un poco de compile en `main` |
| [`= old/first_approach_old4`](= old/first_approach_old4/) | 3 P | Mismo *Observed CVRP names* | Primera eval GBFS seria; `new()` / cargo |
| [`= old/first_approach_old5`](= old/first_approach_old5/) | 3 P | Bloque *Numeric names* (capacity) | OpenAI `_01`/`_02` ya buscan; DeepSeek `h=0` |
| [`= old/first_approach_old6`](= old/first_approach_old6/) | 3 P | Igual que old5 (2ª generación) | Azar: OpenAI `_00`/`_01` no compilan |
| `prompts/` ahora | — | old5 + parche rustc | Aún sin tanda nueva |

---

## Cómo se rompe PlanForge (tres fallos distintos)

Conviene no mezclarlos. Han salido en tandas distintas.

### A. `capacity()` no está (o no es usable)

En el PDDL `capacity`, `demand` y `max_vehicles` **nunca se asignan**. Solo se leen en comparaciones (`(load) + (demand ?to) <= (capacity)`). El traductor puede plegar eso en un `PNE derived!…` y **tirar** la variable `capacity()`. Entonces:

- buscar `name == "capacity()"` puede no encontrar nada;
- buscar `name.contains("capacity")` puede enganchar un `derived!` que **no** es la capacidad (un número plausible y malo).

`load()` y `vehicles()` sí sobreviven: los operadores hacen `increase` / `assign` sobre ellos.

Si `new()` hace `Err` porque no hay `capacity()`, no hay búsqueda. Eso es old4 OpenAI `_01`/`_02`.

### B. `fill_state` / iterar operadores (Rust de la API)

`state.fill_state(registry, &mut props)` devuelve `()`, no `Result`. Encadenar `.map_err(...)` no compila.

`task.get_operators()` ya es `&Vec<Operator>`. `for op in &ops` cuando `ops` ya es esa referencia da `&&Vec`, que no itera. Hay que hacer `for op in ops` o `ops.iter()`.

Eso es DeepSeek en old4 (los 4) y otra vez `_02` DeepSeek en old6.

### C. Hechos proposicionales (`visited` / `loc`)

`get_fact_name` nombra **los dos** lados del dominio: `Atom visited(c7)` y `NegatedAtom visited(c7)`. El valor `0` no significa “cierto”. Cada cliente suele ser **su propia** variable SAS.

Si el código:

1. parsea cualquier nombre que contenga `visited(` (Atom y NegatedAtom),
2. se queda con el **último** valor del dominio (casi siempre el negado),
3. y usa **una sola** `visited_var` para todos los clientes,

en el estado inicial todos “están visitados”, el vehículo está en el depot, y `compute_heuristic` hace `return Ok(0.0)`. GBFS va a ciegas. El log pone `Initial heuristic value for llm_001: 0` y no vuelve a bajar `h`.

OpenAI que funciona solo acepta el prefijo `Atom visited(` / `Atom loc(` y guarda `(var, value)` **por** objeto. El prompt ya dice “no asumas que 0 es true” y enseña `Atom …` e `is_hold`, pero **no prohíbe** parsear `NegatedAtom` ni el `visited_var` global. Eso **aún no está escrito** en `prompts/`.

### D. Rust genérico (no es PlanForge)

`HeuristicBuildError` se construye desde `String`. `.to_string().into()` en un `ok_or_else(|| …)` sin anotar el tipo: rustc ve dos `From` y no compila.

`let mut x = 0.0; x.max(y)`: `0.0` es `{float}` (`f32` o `f64`). Hay que poner `: f64`.

old6 OpenAI `_00` y `_01`. El prompt de old5/old6 **enseñaba** `.to_string().into()`, que es el patrón del primer error.

---

## 1. [`first_approach_old`](= old/first_approach_old/)

Primera pasada. El LLM ve **A-n32-k5** y **A-n37-k5** (n≈32–37, flota 5, capacidad ~100). Demasiado grandes para GBFS a 5 min; aquí casi no hay métricas de búsqueda, sí `INSIGHTS.md` cualitativo.

El prompt ya distingue:

- `variable.name()` → `var0`, `var1`… inútil para semántica;
- `task.get_fact_name(&ExplicitFact::propositional(var, value))` → `Atom loc(c10)`, `Atom visited(c7)`.

Aún no hay el recuadro corto “estos PNE se llaman así / estos otros se pliegan”. El modelo tiene que inferir la API del listado de símbolos y del template Rust.

## 2. [`first_approach_old3`](= old/first_approach_old3/)

Siguen las dos A. Se añade **Observed CVRP names (information, not a recipe)**. Idea: avisar de lo que se *observa* tras traducir, no dar una receta de lookup.

Texto (resumido):

- Numéricos frecuentes: `PNE load()`, `PNE vehicles()`, `PNE total-cost()`.
- `capacity`, `demand`, `travel-cost`, `max_vehicles` a menudo son `PNE derived!…` y **dejan de** llamarse `capacity()` / `demand(c7)`.
- Si un nombre exacto no está: no inventar métodos. *Preferible* `new()` → `Ok` antes que abortar.

“Preferible” no es MUST. `_01` y `_02` ya obligan a un `h` con componente de capacidad / recursos. El modelo puede leer “necesito capacity” + “a veces no está” y decidir abortar si no la encuentra. old3 no lo midió bien en las 8 (casi solo cualitativo).

También aparece la frase de que `HeuristicBuildError` se arma con `.to_string().into()`, no con `&str`. Eso viene de un error de tipos anterior; más tarde se vuelve en contra (old6).

## 3. [`first_approach_old4`](= old/first_approach_old4/)

**Cambio de instancias, no de la historia de fluents.** El split P pequeño: easy `P-n16-k8`, medium `P-n20-k2`, hard `P-n23-k8`. Mismo *Observed CVRP names*. Primera vez que se corre GBFS 300s contra esas tres + builtins. `INSIGHTS.md` mezcla diseño y CSV.

Qué pasó:

| Quién | Compila | Búsqueda |
|---|---|---|
| `main` / `_00` OpenAI | sí | n16 y n20 resuelven (costes 511 / 296 y 537 / 326). n23 timeout |
| `_01` OpenAI | sí | `new()`: *recovered invalid capacity or max_vehicles* |
| `_02` OpenAI | sí | `new()`: *could not recover capacity()* |
| 4 DeepSeek | no | `fill_state().map_err` o `for op in &ops` |

`_01`/`_02` OpenAI hacían lo que el MUST pedía (término de capacidad) y lo que el párrafo vago no conseguía impedir: lookup agresivo + `Err`. En diseño cualitativo quedaban arriba; en máquina no evaluaban ningún estado.

Conclusión de esta tanda: el aviso “a veces es derived, preferible no abortar” **no basta** si el checklist pide `h_capacity` y `capacity()` no existe como PNE.

## 4. [`first_approach_old5`](= old/first_approach_old5/)

Primera generación con el bloque nuevo, en **los 4** templates. Título: **Numeric names: `load`/`vehicles` are safe, `capacity`/`demand`/`max_vehicles` are not**. Sustituye a Observed CVRP names.

Se explica el mecanismo (no solo “a veces pasa”):

1. Los operadores **escriben** `load` / `vehicles` → el traductor conserva esas variables.
2. `capacity` / `demand` / `max_vehicles` solo se leen en `(load)+(demand)≤(capacity)` → el traductor puede fusionar la comparación y **eliminar** `capacity()`.
3. No encontrar `capacity()` es **esperado**. `new()` **no** puede fallar por eso.
4. `contains` / `ends_with` / `starts_with` es peor que no encontrar nada: un `derived!` puede llevar el texto `capacity` y devolver otra cosa.
5. Si no está: `f64::MAX` (capacidad / max vehículos), `0.0` (demanda). Se degrada el término de recursos; `h_routing` sigue.
6. En `_01`/`_02`, MUST NOT explícito: `Err` por esos tres nombres.

`HeuristicBuildError` sigue siendo la frase corta `.to_string().into()`.

**Efecto (lo que se quería):** OpenAI `_01` y `_02` construyen y buscan. En n16 `_01` OA fue rápida (≈0.55s, coste 490). En n23 bajó `h` hasta ~8 (nadie cierra en 300s). `_02` OA fiel al prompt, costes peores donde sí resuelve.

**Efecto colateral (DeepSeek `main`):** compiló (ya no el `map_err` de old4). Logs: `Initial heuristic value: 0` en n16 y n20. n16 “resuelve” coste 450 en 8s con 2.7M expansiones — es GBFS casi ciego, no una heurística buena (`blind` sacó 468). n20 se corta (~14 GB, timeout). Causa: fallo C (`NegatedAtom` + un `visited_var`). El resto de DeepSeek no se llegó a correr del todo en esa pasada.

**No se tocó el prompt** para hechos proposicionales. Solo capacity.

## 5. [`first_approach_old6`](= old/first_approach_old6/)

Segunda generación **con el mismo prompt que old5**. Pregunta: ¿el ranking de old5 era suerte?

CSV (`metrics.csv` de old6):

- Compilan y buscan: `main` OA, `main` DS, `_01` DS, `_02` OA.
- No compilan: `_00` OA, `_00` DS, `_01` OA, `_02` DS.

OpenAI `_00`: `ok_or_else(|| "…".to_string().into())` — rustc E0283, tipo de `HeuristicBuildError` ambiguo.

OpenAI `_01`: `largest_link.max(link)` con `largest_link` inferido como `{float}` — E0689.

DeepSeek `_00`: borrow checker (`scratch` mut dos veces). `_02`: otra vez `fill_state().map_err`.

`main` / `_02` OpenAI sí corren (n16/n20 ok, n23 timeout). `main` DeepSeek aquí **sí da h distintos de 0** y resuelve n16 (471) y n20 (325) — no es el mismo bug que old5; el modelo escribió otro `.rs`.

Lección: el bloque de capacity **se mantiene** (ya no mueren en `new()` por `capacity()`). El ruido ahora es **compilar Rust**, y parte viene de la propia frase del prompt sobre `.into()`.

## 6. `prompts/` actuales (sin generación nueva)

Tras old6 se editaron los 4 templates. Sustituye la frase corta de `HeuristicBuildError` por:

```rust
return Err(HeuristicBuildError::from("missing depot loc fact".to_string()));
.ok_or_else(|| -> HeuristicBuildError { "missing depot loc fact".to_string().into() })?;
```

y: valores de `h` son `f64`; `let mut x: f64 = 0.0` (o `0.0_f64`) antes de `.max` / `.min`.

La siguiente tanda debería nacer de estos `prompts/`. Aún **no** está escrito:

- parsear solo `Atom pred(`, nunca `NegatedAtom`;
- un par `(var, value)` por cliente / sitio, no un `visited_var` global;
- `fill_state` devuelve `()`; no hacerle `map_err`.

---

## Cómo leer esto la próxima vez

1. Si `new()` aborta por `capacity` → el prompt de fluents (old4 vs old5).
2. Si `cargo` habla de `map_err` en `()` o `&&Vec` → API PlanForge (DeepSeek, fallo B).
3. Si el log dice `llm_001: 0` al inicio y no baja → hechos `visited` (fallo C, old5 DeepSeek `main`).
4. Si `cargo` habla de `Into<_>` o `{float}` → Rust del LLM + frase `.into()` (old6; ya parcheado en `prompts/`).

`INSIGHTS.md` de cada carpeta (si existe) es la lectura cualitativa + CSV de **esa** tanda. Este archivo es solo la línea de cambios del prompt.
