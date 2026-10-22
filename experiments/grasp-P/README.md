# GRASP P

Variante RCL-inicial: primer cliente y resto usan el mismo score y RCL, tanto en baseline como en candidatos. Los scores se generaron con un prompt de semilla farthest y se reutilizan intactos, sin nueva llamada. Prompts y respuestas históricas describen aquella generación. Las mediciones anteriores están en `../= old/grasp-farthest-results/`; esta evaluación empieza desde cero y no presupone una mejora.

Baseline y diez candidatos sobre P-n16-k8, P-n23-k8, P-n50-k10, P-n76-k4 y P-n101-k4: 55 ensayos. Seed 1, alpha 0.3, 300 s por ensayo y timeout externo 330 s.

`metrics.csv` conserva primera/mejor solución, tiempos, iteraciones y rutas. `ranking.csv` ordena por cobertura, gap relativo, tiempo a la mejor solución e identificador; solo puede ganar un candidato generado. El ranking es parcial hasta completar los 55 ensayos.

`logs/` contiene salida completa, mejoras y una línea de validación por candidato; `plans/`, las soluciones. `01/`…`10/` contienen los originales `prompt.md`, `out.py` y `out.json` (respuesta de generación).

Ejecutar `experiments/run-grasp.ps1` para E → P → evaluación final. Al pausar, detener el coordinador y su árbol de procesos; al relanzar se conservan los ensayos registrados en CSV y se repite únicamente el ensayo interrumpido. Los hashes de los logs comprueban código, datos y parámetros antes de reutilizar resultados.
