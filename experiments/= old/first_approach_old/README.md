# first_approach

Primera tanda de heurísticas `h(s)` generadas antes de que existiera el pipeline experimental (`experiment.py`, `data/splits/...`). Se conserva como referencia y se documenta aquí cómo se generó y cómo se está probando ahora, sin mezclarlo con el pipeline nuevo.

## Cómo se generó

- **Prompts**: 4 variantes de plantilla, cada una lanzada a 2 proveedores/modelos (`openai / gpt-5.6-sol / medium` y `deepseek / deepseek-v4-pro / medium`): `main_prompt/`, `main_prompt_enhanced_00/`, `main_prompt_enhanced_01/`, `main_prompt_enhanced_02/`. En total 8 candidatos, cada uno con `out - <proveedor> - <modelo> - <effort>.rs` (código), su `.json` (what/why) y su `prompt - ....md` (prompt exacto enviado).
- **Instancias vistas por el LLM (family A)**: únicamente
  - `A-n32-k5`
  - `A-n37-k5`

  Estas dos son las **únicas** que no deben usarse para evaluar: el LLM las vio explícitamente en el prompt.
- Todos los candidatos conservan el nombre de heurística `llm_001` (no se ha usado el renombrado determinista de `generate.rename_heuristic`, porque esta tanda es anterior a esa utilidad).

## Cómo se está probando ahora

Sin ranking (no hay DEV/TEST, no hay selección): solo se ejecuta cada `.rs` contra instancias de A distintas a las del prompt y se compara con heurísticas nativas de PlanForge. Script: [`scripts/try_first_approach.py`](../../scripts/try_first_approach.py).

### Instancias de prueba

Instancias de A no vistas en el prompt, elegidas de tamaño similar a `A-n32-k5`/`A-n37-k5` (30-38 clientes) para que sigan siendo grondeables con los límites por defecto de PlanForge:

```
A-n33-k5, A-n33-k6, A-n34-k5, A-n36-k5, A-n37-k6, A-n38-k5
```

Los splits `dev`/`test` que ya existen en `data/splits/A/` (`A-n45-k6`, `A-n63-k10`, `A-n39-k5`, `A-n54-k7`, `A-n69-k9`) **no se usan aquí**: son instancias más grandes y el grounder por defecto de PlanForge revienta con ellas (`grounding exceeded the atom limit: ... (limit 10,000,000)`) incluso con la heurística `blind`, es decir, es un problema de grounding, no de la heurística. Para probarlas haría falta subir `--max-ground-atoms` (ver más abajo), a costa de mucha más memoria/tiempo.

### Heurísticas nativas de PlanForge usadas como referencia

De todo lo que registra `planforge-search/src/heuristic_factory.rs`, se usan las que no necesitan argumentos ni CPLEX y aplican a cualquier tarea (`requirements: any_task`):

- `blind`
- `ff`
- `lmcutnumeric`

Se descartan explícitamente: `posthoc_optimization`, `pot_da_ocp`, `numeric_potential` (requieren CPLEX); `max`, `canonical`, `scp`/`cost_partitioning`, `greedy_numeric_pdb`, `canonical_numeric_pdb`, `max_numeric_pdb` (necesitan fuentes de abstracción/PDB como argumento, no se pueden lanzar "tal cual"); `domain_abstraction`, `cartesian_abstraction`, `canonical_domain_abstractions`, `multi_domain_abstractions` (construyen una abstracción CEGAR por instancia, mucho más caras — se pueden añadir a `BUILTIN_HEURISTICS` en el script si se quiere ampliar la comparación).

### Cómo ejecutarlo

`benchmark.py` compila contra el crate real de PlanForge (`experiments/llm-cvrp-runner` en el repo hermano `mulcia-tfm-PlanForge`), y ese repo **solo compila en Linux** (usa `libc::clock_gettime`, no disponible en Windows). Si se trabaja desde Windows, hay que lanzarlo en WSL:

```bash
wsl -e bash -lc "cd /mnt/c/Code/GitHub/TFM/mulcia-tfm-hllm-pipeline && uv run python scripts/try_first_approach.py"
```

El script:
1. Compila el binario base `planforge` una vez y corre `blind`/`ff`/`lmcutnumeric` contra las 6 instancias.
2. Para cada `.rs` de `first_approach/*/out - *.rs`: lo escribe como `main.rs` del runner, compila, y si compila lo corre contra las 6 instancias (si no compila, se salta y se deja constancia en el log).
3. Guarda todo en `quick_test_metrics.csv` (en esta misma carpeta; el script no sobrescribe si ya existe, hay que borrarlo antes de relanzar).

Límite de tiempo por ejecución: `MAX_TIME_S = 20` en el script (pensado para una comparación rápida, no exhaustiva; se puede subir si se quiere dejar correr más).

## Resultado de la primera ejecución (13/09/2026)

- **Los 8 candidatos fallan al compilar** contra el checkout actual de `mulcia-tfm-PlanForge`. Los 8 hacen `use planforge_sas::task::ExplicitFact;`, pero ese módulo no existe: `ExplicitFact` vive en `planforge_sas::numeric_task` (el mismo módulo que ya usan correctamente para `AbstractNumericTask`/`TaskRef`). Es decir, no es un problema de versión del repo (no hay commits de `planforge-sas`/`planforge-search` entre la generación y ahora): el LLM alucinó un módulo `task` que no existe, de forma consistente en las 8 generaciones. Cada `.rs` tiene además errores adicionales propios (métodos inexistentes como `NumericVariable::id()`, `ExplicitVariable::values()`, `HeuristicBuildError: From<&str>` no implementado, etc.), así que corregir el import no bastaría para que compilen sin más ajustes.
- Con `MAX_TIME_S = 20`, ninguna de las 3 heurísticas nativas resuelve ninguna de las 6 instancias en el tiempo dado (CVRP con astar/blind-ff-lmcutnumeric es duro incluso a esta escala); sirven de referencia de "no resuelto en 20s", no de coste de solución. Subir `MAX_TIME_S` da una comparación más justa a costa de tardar más en total.
- CSV con el detalle: [`quick_test_metrics.csv`](./quick_test_metrics.csv).

### Próximo paso si se quiere seguir con esta tanda

Corregir a mano el import (`planforge_sas::task` → `planforge_sas::numeric_task`) y los demás errores de compilación en cada `.rs`, o regenerar candidatos nuevos con el pipeline actual (`experiment.py`), que usa la misma plantilla (`templates/heuristic_template.rs`, con el import correcto) pero no garantiza que el LLM no vuelva a alucinar la API.
