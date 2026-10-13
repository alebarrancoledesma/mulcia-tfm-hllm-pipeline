# Datos del pipeline

Dos familias oficiales para el experimento de transferencia: **P** y **E**.
Mismo `processed/pddl/domain.pddl`.

## Por qué no A/B (ni P grande)

El README de CVRPLIB (`mulcia-tfm-artifacts/data/cvrplib/README.md`) deja claro que **P reutiliza instancias de A, B y E** (cambia capacidad / número de rutas). Por eso:

- **Augerat A** (mínimo n=32) y **Augerat B** (mínimo n=31) son demasiado grandes para GBFS/FF a 5 min, y además están en la genealogía de P.
- Los P grandes (`P-n50`, `P-n76`, `P-n101`) son reparametrizaciones de `E-n51` / `E-n76` / `E-n101`: **no se usan en los splits**.

## Familias oficiales

| Carpeta | Set CVRPLIB | Qué es | Splits (solo n pequeña) |
|---|---|---|---|
| `processed/pddl/P/` | Augerat **P** | Capacidad reparametrizada; la banda n=16–23 es asequible para un planner | prompt/dev/test en `splits/P/` |
| `processed/pddl/E/` | Christofides–Eilon **E** | Otra fuente histórica (1969); coords distintas de los P pequeños | prompt/dev/test en `splits/E/` |

Los P-n16…P-n23 y los E-n13…E-n33 **no comparten el mismo grafo**. El cruce P↔E del README afecta a las instancias grandes, que quedan fuera del protocolo.

No hay carpetas A/B en este repo.

## Splits

```
P/prompt: P-n16-k8, P-n20-k2, P-n23-k8
P/dev:    P-n16-k8, P-n19-k2, P-n20-k2, P-n22-k8
P/test:   P-n21-k2, P-n22-k2

E/prompt: E-n13-k4, E-n22-k4, E-n30-k3
E/dev:    E-n13-k4, E-n22-k4, E-n23-k3, E-n30-k3
E/test:   E-n31-k7, E-n33-k4
```

Mismo recuento en ambas familias: 3 prompt (fácil / media / difícil), 4 dev (ranking), 2 test. En E solo hay 6 instancias n≤33; por eso `E-n30-k3` entra en prompt y en dev, y el test se queda con las dos que sobran.
