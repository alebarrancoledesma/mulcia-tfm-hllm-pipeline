# Experimentos GRASP

El lanzador es `experiments/run-grasp.ps1`. Ejecuta la selección de E, la de P
y, cuando ambas terminan, la evaluación final de las dos ganadoras y el básico.
Cada ensayo tiene 300 segundos de búsqueda, `alpha=0.3` y semilla 1.
La misma heurística y RCL se utilizan para elegir el primer cliente y las
inserciones posteriores. La búsqueda local es 2-opt dentro de cada ruta.

## Validación, fallos y reanudación

La validación comprueba AST, llamadas por nombre de parámetro, valores finitos,
ausencia de mutación y determinismo dentro de cada proceso y entre dos procesos.
Incluye rutas vacías, capacidad exacta y clientes de las cinco instancias de
selección. La integración con el motor se prueba en una instancia sintética
pequeña: no encontrar solución dentro del plazo del smoke es solo diagnóstico,
no invalida globalmente una heurística. Un timeout externo de validación queda
como comprobación inconclusa y permite el ensayo real; una excepción o violación
del contrato sí lo invalida. El watchdog de validación es de 20 s por proceso.

Cada ensayo real dispone de 300 s (watchdog externo de 330 s). Los fallos guardan
una fila con el mismo encabezado, `solved=False`, estado `timeout`, `error` o
`invalid`, y mediciones desconocidas vacías. No se inventan ceros. Los mensajes
del CSV son breves, de una línea, y remiten a `logs/`; las trazas y advertencias
completas se conservan allí. Los fallos cuentan como cobertura cero. El básico
se evalúa sobre las mismas cinco instancias que cada candidato de su familia.

La corrección del evaluador conserva las 55 filas correctas de E y las 20 de P,
incluidos sus textos originales de cada campo CSV. Las siete falsas exclusiones
de P ya retiradas quedan pendientes de ensayo real. La migración añade
`RESUME_CONFIG <firma> evaluator-only ...` a cada log completado, sin sustituir
su `CONFIG` histórico, después de verificar las firmas previas y las huellas
sin cambios de candidatos, básico, motor, entradas y parámetros. La firma nueva
sigue incluyendo el evaluador. No se admite un cambio posterior de esas fuentes.
Los rankings permanecen parciales hasta completar los 55 ensayos de cada familia;
la evaluación final verifica independientemente ambos CSV y rankings.

## Selección y ranking

`grasp_evaluate.py`, función `ranking`, ordena los candidatos de cada familia
de forma lexicográfica, en este orden:

1. **Mayor cobertura** (`coverage`): número de instancias con solución factible
   dividido entre las cinco instancias de selección.
2. **Menor desviación relativa media** (`mean_relative_gap`): calidad de la mejor
   solución encontrada dentro de los 300 segundos, normalizada por instancia.
3. **Menor tiempo medio hasta la mejor solución** (`mean_time_to_best`): desempate
   cuando coinciden la cobertura y la desviación relativa media.
4. Si también coincide el tiempo, **identificador del candidato en orden
   lexicográfico**, únicamente para obtener un orden reproducible.

Para una instancia i, la referencia b_i es el menor coste final factible observado
entre todos los candidatos de esa familia y el básico. Para el candidato h:

```text
gap(h, i) = (coste(h, i) - b_i) / b_i
mean_relative_gap(h) = media de gap(h, i) en las instancias que h resuelve
```

El CSV guarda el gap como fracción: 0.01 equivale a un 1 %. La referencia es el
**mejor resultado observado en esta comparación**, no el óptimo de CVRPLIB.
Así no se suman directamente costes de instancias con escalas diferentes.
Las distancias del motor son euclídeas sin redondear.

El tiempo se promedia también solo sobre las instancias resueltas. Si dos
candidatos resuelven distintos subconjuntos con la misma cobertura, sus medias
se calculan sobre esos subconjuntos; los fallos no reciben un coste inventado.
Sin soluciones, las medias quedan vacías y se ordenan como infinito.
Se comparan los valores numéricos completos, sin redondearlos ni aplicar una
tolerancia de empate.

`time_to_best` mide cuándo se encontró la mejor solución conservada, desde el
inicio de la búsqueda. **No es `runtime`**, que normalmente ronda los 300 s,
ni el tiempo de la primera solución. `first_solution_time` y
`first_solution_cost` describen la primera construcción factible, antes de
2-opt, y no intervienen en el ranking principal. Las iteraciones tampoco son
un criterio de desempate.

Cada familia guarda `metrics.csv` y `ranking.csv` en su carpeta. El ranking
es parcial hasta completar los 55 ensayos: 10 candidatos y el básico sobre
cinco instancias. Las referencias y posiciones pueden cambiar mientras falten
resultados. El básico aparece en la clasificación, pero la ganadora seleccionada
es la primera **heurística generada** con cobertura positiva. Si ninguna resuelve
un caso, no se selecciona ganadora.

## Ejemplo del desempate

En E-n23-k3, E-02 y el básico alcanzaron el mismo coste final, 596.911711983.
E-02 lo alcanzó en 3.715 s y el básico en 13.718 s: para ese caso, E-02 gana
el desempate temporal. Esto no determina por sí solo el ranking de la familia,
que agrega sus cinco instancias. Una mejora de calidad tiene prioridad sobre
el tiempo: E-01 consiguió 594.079171747, aunque tardó más en alcanzarlo.

## Evaluación final

Las ganadoras se fijan con sus rankings de selección. Después se ejecutan ambas
y el básico sobre E-n30-k3, E-n22-k4, P-n19-k2 y P-n40-k5, sin usarlos para
reelegir las ganadoras. E-n22-k4 tuvo ejecuciones exploratorias previas y no se
presenta como un caso completamente intacto. Los cinco ejemplos de cada prompt
se utilizan para selección, no para estimar generalización.

Se realiza una ejecución por combinación con semilla 1. Las diferencias de
tiempo son descriptivas de estas ejecuciones, no evidencia estadística basada
en repeticiones.

## Resumen para la memoria

> Los candidatos se ordenan por cobertura de soluciones factibles, seguida de
> la desviación relativa media del coste final respecto al mejor coste observado
> en cada instancia. Los empates se resuelven mediante el tiempo medio hasta
> alcanzar la mejor solución. Se selecciona una heurística generada por familia
> tras completar todos los ensayos de selección; el método básico se conserva
> como referencia y la evaluación final no interviene en dicha selección.
