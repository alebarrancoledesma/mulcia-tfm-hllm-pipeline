# Generación y evaluación de heurísticas

Código experimental para generar funciones heurísticas con un LLM, validarlas y compararlas en planificación numérica y GRASP. Los datos y resultados publicados se encuentran en el repositorio hermano `mulcia-tfm-artifacts`.

## Requisitos

- Python 3.10 o posterior y `uv`.
- Una variable `OPENAI_KEY` en `.env` para generar candidatos.
- `mulcia-tfm-artifacts`, `mulcia-tfm-grasp` y `mulcia-tfm-PlanForge` como directorios hermanos. Las rutas también pueden pasarse a `run.ps1`.
- WSL, Rust y Cargo para la evaluación con PlanForge.

```powershell
uv sync
Copy-Item .env.example .env
```

Después se completa `OPENAI_KEY` en el `.env` local. Este archivo está ignorado por Git y no debe publicarse. La plantilla también incluye las variables opcionales `DEEPSEEK_API_KEY` y `ARTIFICIAL_ANALYSIS_KEY`, utilizadas únicamente por funciones auxiliares asociadas a esos servicios.

## Lanzador

Todas las ejecuciones nuevas se escriben en `outputs/`, que no se versiona.

```powershell
# Planificación numérica
.\run.ps1 -Experiment numeric -Stage generate -Family E
.\run.ps1 -Experiment numeric -Stage evaluate -Family E

# GRASP
.\run.ps1 -Experiment grasp -Stage generate -Family E
.\run.ps1 -Experiment grasp -Stage evaluate -Family E
.\run.ps1 -Experiment grasp -Stage cross
```

Se repiten las mismas órdenes con `-Family P`. La evaluación numérica compila cada candidato dentro del runner de PlanForge y restaura el archivo original al terminar. La evaluación GRASP valida primero el contrato de cada función y puede reanudar ensayos completos.

## Estructura

```text
prompts/      plantillas de instrucciones
templates/    interfaces de Rust y Python
scripts/      generación, validación y evaluación
src/          acceso al LLM, datos y ranking
outputs/      ejecuciones locales
```

Los resultados utilizados en la memoria no se regeneran ni se modifican desde este repositorio; su copia canónica está en `mulcia-tfm-artifacts/outputs`.
