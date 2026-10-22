# Evaluación de heurísticas (DEV)

Se cruzan **E_09 (hE)** y **P_07 (hP)** con tres casos: **E-n13-k4,
P-n16-k8 y P-n19-k2**. Son seis ejecuciones secuenciales, una por combinación.
E/09 y P/07 ocupan el primer puesto de sus `ranking.csv` originales.
Se eligió el único caso E resuelto y los dos casos P más pequeños resueltos.
Las fuentes son `experiment-E/09/out.rs` y `experiment-P/07/out.rs`.

## Resultados guardados

La ejecución en WSL Ubuntu-24.04 terminó con dos compilaciones y seis ensayos
con exit 0, sin timeouts, y produjo seis planes. `main.rs` quedó restaurado.

| Candidato | Instancia | Resuelto | Search time (s) | Coste |
|---|---|---|---:|---:|
| E_09 | E-n13-k4 | yes | 0.014319 | 327 |
| P_07 | E-n13-k4 | yes | 0.059039 | 340 |
| P_07 | P-n16-k8 | yes | 2.555990 | 511 |
| E_09 | P-n16-k8 | yes | 3.901925 | 564 |
| P_07 | P-n19-k2 | yes | 4.644958 | 269 |
| E_09 | P-n19-k2 | yes | 5.688354 | 394 |

`results/` contiene `metrics.csv`, `metrics-target-E.csv` y `metrics-target-P.csv`.
Las columnas son `candidate,instance,solved,runtime,expanded,evaluated,generated,cost`.
Se ordenan por tiempo de búsqueda ascendente, con los no resueltos al final.
`runtime` es Search time, no tiempo total; los datos ausentes quedan vacíos.
`solved=yes` indica que el log contiene `Solution found!`.
`logs/` guarda las dos compilaciones y los seis logs completos; `plans/`, los planes.
Cada ensayo se identifica como `candidato__instancia.log` o `.sas_plan`.

## Reproducción desde PowerShell

```powershell
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Code/GitHub/TFM/mulcia-tfm-hllm-pipeline/experiments/evaluations/compile_and_run.sh /tmp/evaluacion-E-P-nueva
```

La salida debe estar libre de `results/`, `logs/` y `plans/`; sin argumento se usa
esta carpeta y se rechaza sobrescribir los resultados guardados.
PlanForge debe estar en el repositorio hermano `mulcia-tfm-PlanForge`.
Se añade `~/.cargo/bin` al PATH y se compila una vez por candidato con
`cargo build --release --locked --offline -p planforge-llm-cvrp`.
La búsqueda es `gbfs(llm_001())`, con los límites realmente usados:
`--max-time 60s --max-memory 4G` y `timeout --verbose --kill-after=5s 90s`.
Cada ensayo usa un directorio temporal propio para evitar planes anteriores.
Un backup y `trap` restauran `main.rs`; `flock` usa `.heuristic-evaluation.lock`.
Se comparte `PlanForge/target`; el binario queda del último candidato compilado.
Los exits se anotan en los logs; un fallo de compilación detiene el script.
Solo se usa Python estándar para los CSV; no se llama a LLM ni se lee `.env`.

## Conclusión

hE obtiene menor tiempo y coste en el caso E, y hP en los dos casos P.
En esta muestra, cada heurística funciona mejor en la familia utilizada en sus
prompts, lo que apunta a una especialización y no a una mejora uniforme sobre
cualquier instancia. El patrón también se mantiene en P-n19-k2, que no aparecía
en ninguno de los prompts. Aun así, el número de casos es pequeño y solo se ha
realizado una ejecución por combinación, por lo que el resultado debe tomarse
como un indicio, no como una demostración general.

## Limitaciones

Es DEV, con selección basada en resultados previos y una sola repetición.
E-n13-k4 aparece en el prompt de hE y P-n16-k8 en el de hP;
P-n19-k2 no aparece en ninguno. No es un test independiente.
Tres casos pequeños no permiten concluir generalización.
GBFS no garantiza optimalidad y no se ejecutó un validador externo de planes.
