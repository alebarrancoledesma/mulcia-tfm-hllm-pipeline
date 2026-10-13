# first_approach — Insights consolidados

Análisis de las 4 tandas de generación con los prompts actuales (old5 → esta), con foco en qué compila, qué guía la búsqueda, y qué template y modelo es más fiable.

Tandas cubiertas:

| Carpeta | Alias al generar | Prompt base |
|---|---|---|
| `= old/first_approach_old5` | `first_approach` / `first_approach_01` | Numeric names (capacity) |
| `= old/first_approach_old6` | `first_approach_02` | Igual que old5 |
| `experiments/first_approach` | — | old5 + parche rustc (`HeuristicBuildError::from`, `f64`) |

old4 usaba prompts más viejos (Observed names, instancias A); se menciona donde es relevante como referencia.

---

## Resultados por tanda — OpenAI `gpt-5.6-sol`

| Template | old5 | old6 | esta | Compila |
|---|---|---|---|---|
| `main_prompt` | ✅ n16+n20 (537/326) | ✅ n16+n20 (537/326) | ❌ E0308 | 2/3 |
| `_00` | ✅ n16+n20 (537/326) | ❌ E0283 | ✅ n16+n20 (537/326) | 2/3 |
| `_01` | ✅ n16+n20 (490/451) | ❌ E0689 | ✅ n16+n20 (~1ms, 595/491) | 2/3 |
| `_02` | ✅ n16+n20 (548/431) | ✅ n16+n20 (531/436) | ❌ E0599 | 2/3 |

**OpenAI compila en ~2/3 de las tandas por template.** No hay ninguno que falle siempre ni que tenga éxito siempre. El error varía entre tandas aunque el prompt sea el mismo (temperatura de muestreo).

## Resultados por tanda — DeepSeek `deepseek-v4-pro`

| Template | old5 | old6 | esta | Observación |
|---|---|---|---|---|
| `main_prompt` | ✅ n16 (450), n20 cortado | ✅ n16+n20 (471/325) | ✅ n16+n20 (495/407) | El más fiable de DS |
| `_00` | no corrido | ❌ E0499 scratch | ❌ E0499 scratch | Borrow doble siempre |
| `_01` | no corrido | ✅ n16+n20 (537/326, h=cte?) | ✅ h=32 fijo en n16/n20/n23 | h constante = blind |
| `_02` | no corrido | ❌ E0599 fill_state | ❌ E0502 borrow bucle | No compila |

**DeepSeek es menos estable.** Cuando compila y `h` no es constante, solo `main_prompt` da heurísticas realmente informativas. `_00` cae siempre por el mismo borrow doble en `scratch`. `_01` compila pero produce `h` constante (equivale a búsqueda ciega).

---

## Modos de fallo (no mezclar)

### A. Borrow checker en `scratch` (E0499 / E0502) — DeepSeek `_00` / `_02`

El prompt sugiere `RefCell<ScratchBuffers>` con varios campos. El LLM luego toma dos `&mut` a campos distintos del mismo struct:

```rust
let prop = &mut scratch.prop;
let numeric = &mut scratch.numeric;  // E0499: scratch ya tiene borrow
```

O itera un campo mientras muta otro del mismo struct (`for &c in &sc.unvisited { sc.in_rem[c] = true; }`). El prompt **no indica que dos `&mut` a campos distintos del mismo struct no son válidos** sin destructuring o `split_at_mut`.

### B. Método inventado (E0599) — OpenAI `_02` esta tanda

`fact.variable()` no existe en `ExplicitFact`. El LLM asumió una API convencional que no está documentada. El prompt ya dice "Do NOT invent PlanForge methods", pero lo ignora al azar.

### C. Función sin retorno (E0308) — OpenAI `main` esta tanda

Un punto y coma de más tras `Ok(...)` hace que `compute_heuristic` devuelva `()` en vez de `Result<f64, EvaluationError>`. Es un error trivial de Rust, no de lógica CVRP.

### D. `h` constante — DeepSeek `_01`

`h` devuelve 32 en las tres instancias (n16, n20, n23) desde el estado inicial. No varía con el estado: GBFS expande igual que `blind`. El log muestra `Initial heuristic value for llm_001: 32` y expansiones idénticas a blind (5.2M en n16). Causa probable: parseo de `visited` incorrecto (`NegatedAtom` / variable global) o heurística que no depende de los clientes no visitados reales del estado.

### E. Errores del parche rustc (old6) — resueltos en esta tanda

old6 OpenAI `_00`: E0283 por `.to_string().into()` sin anotar tipo. old6 OpenAI `_01`: E0689 por `0.0.max(...)` con `{float}` ambiguo. El parche en los prompts actuales (`HeuristicBuildError::from(...)` explícito, `let mut x: f64`) eliminó ambos en esta tanda.

---

## Calidad de las heurísticas que sí corren

### `h` inicial en el estado de partida

| Candidato | n16 `h₀` | n20 `h₀` | Qué indica |
|---|---|---|---|
| blind | 6 | 6 | Profundidad restante, sin información CVRP |
| ff | 380 | 511 | Estimación de acciones FF |
| lmcutnumeric | 129 | 151 | Cota inferior exacta |
| OA `main` (old5) | ~300+ | ~400+ | Coste de ruta estimado |
| OA `_00` (todas) | 172 | 189–214 | Ruta + viaje al depot |
| OA `_01` (old5) | — | — | Vecino más cercano + capacidad |
| OA `_01` (esta) | 127 | 148 | Estimación muy informativa, resuelve en ~1ms |
| DS `main` (esta) | 166 | 206 | Informativa, guía real |
| DS `_01` (esta) | **32** | **32** | Constante — equivale a blind |

### Resolución n16 + n20 (costes)

Ninguna heurística ha resuelto n23 en 300s en ninguna tanda. n16 y n20 los resuelven todas las que compilan y tienen `h > 0`.

| Candidato | n16 coste | n20 coste | Expansiones n16 |
|---|---|---|---|
| ff | 540 | 529 | 81K |
| OA `_00` (todas tandas) | **537** | **326** | 893K |
| OA `main` old4 | 511 | 296 | 1.05M |
| OA `_01` old5 | 490 | 451 | 449K |
| OA `_01` esta | 595 | 491 | **58 / 21** |
| OA `_02` old5 | 548 | 431 | 1.25M |
| DS `main` esta | 495 | 407 | 1.89M |
| DS `main` old6 | 471 | 325 | 45K |

OA `_00` da siempre el mismo coste (537/326) y las mismas expansiones (893K n16, 2.96M n20): el código generado es funcionalmente idéntico tanda a tanda. Es el candidato más **reproducible**.

OA `_01` esta tanda resuelve en 58/21 expansiones (prácticamente directo), pero con coste peor (595/491). GBFS greedy, heurística muy dirigida pero no admisible.

---

## Diferencias entre templates

| Template | Qué añade sobre el anterior |
|---|---|
| `main_prompt` | Base: API, no hardcodear, libertad de diseño |
| `_00` | + construcción vs evaluación detallada, scratch buffers con `RefCell`, operadores y efectos numéricos, runtime safety |
| `_01` | + **"Client Instance Family Profile"** (sección explícita de escala, flota, demandas, routing actions) + **MUST reflect at least two** patrones en el código + `h` como suma de ≥2 componentes explícitos + `instance_patterns` en el JSON |
| `_02` | + especialización de cliente aún más fuerte: debe citar qué patrones de las instancias motivaron cada decisión; routing completion MUST depender del estado actual (no orden fijo en `new()`) |

`_00` es más rico en API que `main` pero el diseño sigue siendo libre. `_01` y `_02` son explícitamente **client-based**: obligan al LLM a que el código refleje las características de la familia de instancias (flota pequeña, demanda heterogénea, visit-via-depot, matriz de costes explícita). Esto genera heurísticas más especializadas cuando compilan, pero código más complejo que tropieza más con Rust.

## Qué template usar

`main_prompt` y `_00` son los más estables para OpenAI: generan heurísticas con `h` informativo, compilan 2/3 de las tandas, y el código resultante es más simple (menos borrow checker).

`_01` y `_02` son más ambiciosos en diseño pero generan código más complejo que tropieza más con Rust. Cuando compilan, el resultado puede ser mejor (`_01` old5: 490/451, 449K expansiones) o peor (`_01` esta: 595/491, solo 58 expansiones — coste alto porque GBFS greedy puro).

Para DeepSeek, solo `main_prompt` es fiable: compila en las 3 tandas y produce heurísticas informativas. Los demás templates caen siempre por la misma razón (borrow doble de `scratch` en `_00`, `h` constante en `_01`, borrow en bucle en `_02`).

**Conclusión: `main_prompt_enhanced_00` con OpenAI es el punto de equilibrio** — más contexto que `main_prompt` (construcción vs evaluación, scratch buffers, operadores) sin llegar a requerir diseños tan complejos que rompan el borrow checker.

---

## Conclusión para el experimento de ranking

El modelo más capaz disponible es **OpenAI `gpt-5.6-sol` (reasoning medium)**. Con el prompt más libre (`main_prompt`) genera heurísticas con lógica CVRP completa (capacity, demand, load, visited, loc, matriz de distancias desde operadores) incluso sin que el prompt lo exija explícitamente — lo infiere del dominio PDDL. La tasa de compilación es ~66% por candidato; con 10 candidatos se esperan ~6-7 compilables y funcionalmente distintos entre sí (el diseño libre maximiza la varianza del muestreo). Los prompts más prescriptivos (`_01`, `_02`) pueden dar mejores heurísticas individuales pero con ~25% de compilación y código más frágil.

**Para el experimento de ranking**: generar N candidatos independientes con `main_prompt` + OpenAI `gpt-5.6-sol`, evaluar los que compilen en las 3 instancias P con GBFS 300s, y ordenar por coste/expansiones.

---

## Lo que el prompt no cubre (y sigue rompiendo)

1. **Borrow doble en structs con `RefCell`**: el prompt muestra el patrón pero no advierte que tomar `&mut` a dos campos del mismo struct simultáneamente no compila. Afecta a DeepSeek `_00`/`_02` de forma sistemática.

2. **`Atom` vs `NegatedAtom` en `visited`**: el prompt dice "no asumas que 0 es verdadero" y muestra `Atom loc(c10)`, `Atom visited(c7)`. Pero no prohíbe explícitamente parsear `NegatedAtom visited(c7)` ni usar una variable global `visited_var` para todos los clientes. Cuando el LLM lo hace, `h` queda constante (todos "visitados" en el estado inicial). Afecta a DeepSeek `_01` en todas las tandas donde corre.

3. **No inventar métodos de la API**: el prompt lo dice explícitamente pero OpenAI lo ignora en ~1/4 tandas para `_02` (`.variable()` en `ExplicitFact`).
