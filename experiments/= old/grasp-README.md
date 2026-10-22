# Candidatos GRASP RCL

## Regeneración real

Diez candidatos por familia en `grasp-E/01..10/` y `grasp-P/01..10/`.
Cada candidato contiene exclusivamente `prompt.md`, `out.py` (respuesta original,
sin reparaciones) y `out.json` con las claves `what` y `why`.
Las carpetas de candidatos previas se eliminan, sin archivos ni copias de respaldo.

Cinco ejemplos completos por prompt, procedentes del repositorio hermano
`mulcia-tfm-artifacts/data/cvrplib`, todos EUC_2D:

- E: E-n23-k3, E-n33-k4, E-n51-k5, E-n76-k14, E-n101-k8.
- P: P-n16-k8, P-n23-k8, P-n50-k10, P-n76-k4, P-n101-k4.

La selección cubre tamaños pequeños, medianos y grandes y distintas capacidades
y presiones de carga. E/P76 y E/P101 comparten geometría entre familias, pero
tienen distinto Q; no son geometrías independientes. E-n33-k4 fue test del
experimento HLLM anterior y ahora es ejemplo RCL. Se conservan coordenadas,
demandas y capacidad completas; se omiten únicamente COMMENT con optimal/best
value y no se suministran soluciones `.sol`.

Generador: `scripts/grasp_gen_candidates.py`, wrapper `openai_generate`, proveedor
OpenAI, modelo `gpt-5.6-sol`, reasoning effort `medium`, schema strict
`code/what/why`, 3 workers, 6 solicitudes/minuto, timeout 300 s y max_retries 0.
La generación comprueba sintaxis y firma mediante AST sin ejecutar el código.
Los metadatos del modelo se documentan aquí, no en los JSON de candidatos.

## Evaluación pendiente

No se ejecuta evaluación durante la generación. DEV E: E-n22-k4, E-n30-k3;
DEV P: P-n19-k2, P-n40-k5. Las cuatro instancias quedan fuera de todos los
prompts. DEV sirve para selección, no como estimación independiente de
generalización.

El evaluador `scripts/grasp_evaluate.py` carga `out.py`. Una ejecución explícita
posterior contempla 44 ensayos DEV (baseline y diez candidatos por instancia)
y cuatro cruces, seed 1, alpha 0.3, 300 s/ensayo y watchdog de 330 s.
Usa distancias euclídeas del motor, sin comparar con óptimos CVRPLIB.
