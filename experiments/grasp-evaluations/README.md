# GRASP: selección y evaluación final
Variante actual: RCL desde ruta vacía, con idéntico constructor para baseline y los 20 candidatos. Scores generados bajo prompt farthest, reutilizados intactos sin nueva llamada. Resultados anteriores separados en `../= old/grasp-farthest-results/` (5 ensayos E, 0 P); reinicio desde cero sin inferir mejoras.
Desde la raíz: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\run-grasp.ps1`.
Etapas secuenciales: E (55 ensayos en sus 5 ejemplos del prompt), P (55 en los suyos), final (12: baseline y ambas ganadoras sobre 4 casos finales); se detiene ante error.
Parámetros: 300 s, seed=1, alpha=0.3, timeout externo=330 s. Se reutilizan los ensayos completos del CSV tras comprobar hashes y parámetros; la validación rápida se repite al arrancar.
Selección: `../grasp-E/` y `../grasp-P/`; final: `results/`; logs del lanzamiento en segundo plano: `pipeline.log` y `error.log`; rankings pendientes hasta completar selección.
Los ganadores se fijan en memoria tras verificar ambos rankings completos. E-n22-k4 tuvo ensayos exploratorios anteriores y no se considera un caso inédito.
