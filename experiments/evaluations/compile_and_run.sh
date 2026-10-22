#!/usr/bin/env bash
# E/09 y P/07 sobre tres casos DEV: seis ensayos, una compilación por candidato.
set -Eeuo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
PIPE="$(cd "$HERE/../.." && pwd)"
PF="$(cd "$PIPE/../mulcia-tfm-PlanForge" && pwd)"
MAIN="$PF/experiments/llm-cvrp-runner/src/main.rs"
DOMAIN="$PIPE/data/processed/pddl/domain.pddl"
INSTANCES=(E-n13-k4 P-n16-k8 P-n19-k2)
[[ $# -le 1 ]] || { echo "Uso: bash $0 [directorio-salida]" >&2; exit 1; }
OUT="${1:-$HERE}"
for dir in results logs plans; do
    [[ ! -e "$OUT/$dir" ]] || { echo "Ya existe $OUT/$dir; usa una salida nueva." >&2; exit 1; }
done
for tool in cargo python3 timeout flock; do command -v "$tool" >/dev/null; done
exec 9>"$PF/.heuristic-evaluation.lock"
flock -n 9 || { echo 'Otra evaluación está activa' >&2; exit 1; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
mkdir "$OUT/results" "$OUT/logs" "$OUT/plans"
LOGS="$OUT/logs"
export CARGO_TARGET_DIR="$PF/target"
export CARGO_NET_OFFLINE=true

WORK="$(mktemp -d)"
cp -p "$MAIN" "$WORK/main.rs.before"
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if ! cp -p "$WORK/main.rs.before" "$MAIN"; then
        echo "No se pudo restaurar main.rs; copia disponible en $WORK" >&2
        exit 1
    fi
    rm -rf "$WORK"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

failed=0
for candidate in E_09 P_07; do
    family="${candidate%_*}"
    number="${candidate#*_}"
    cp "$PIPE/experiments/experiment-$family/$number/out.rs" "$MAIN"
    echo ">>> Compilando $candidate"
    rc=0
    (cd "$PF" && cargo build --release --locked --offline -p planforge-llm-cvrp) \
        > "$LOGS/build-$candidate.log" 2>&1 || rc=$?
    echo "Exit compilación: $rc" >> "$LOGS/build-$candidate.log"
    if [[ $rc -ne 0 ]]; then
        echo "Falló la compilación: $LOGS/build-$candidate.log" >&2
        exit "$rc"
    fi
    for inst in "${INSTANCES[@]}"; do
        name="${candidate}__${inst}"
        log="$LOGS/$name.log"
        mkdir "$WORK/$name"
        echo ">>> $name  gbfs(llm_001())"
        rc=0
        (cd "$WORK/$name" && timeout --verbose --kill-after=5s 90s \
            "$PF/target/release/planforge-llm-cvrp" --search 'gbfs(llm_001())' \
            --max-time 60s --max-memory 4G "$DOMAIN" \
            "$PIPE/data/processed/pddl/${inst:0:1}/$inst.pddl") > "$log" 2>&1 || rc=$?
        echo "Exit ensayo: $rc" >> "$log"
        if [[ $rc -ne 0 ]]; then
            echo "Falló $name (exit $rc); consulta $log" >&2
            failed=1
        elif ! grep -q 'Solution found!' "$log"; then
            echo "Sin solución: $name" | tee -a "$log"
        fi
        if [[ -f "$WORK/$name/sas_plan" ]]; then
            cp "$WORK/$name/sas_plan" "$OUT/plans/$name.sas_plan"
        fi
    done
done

# Solo parsing de logs y escritura de CSV, con la biblioteca estándar.
python3 - "$OUT" <<'PY'
import csv
from pathlib import Path
import re
import sys

out = Path(sys.argv[1])
header = 'candidate,instance,solved,runtime,expanded,evaluated,generated,cost'.split(',')
number = r'([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)'
patterns = {'runtime': rf'Search time:\s*{number}s',
            'expanded': r'Expanded\s+(\d+) state',
            'evaluated': r'Evaluated\s+(\d+) state',
            'generated': r'Generated\s+(\d+) state',
            'cost': rf'Plan cost:\s*{number}'}
rows = []
for log in sorted((out / 'logs').glob('*__*.log')):
    candidate, instance = log.stem.split('__')
    text = log.read_text(encoding='utf-8', errors='replace')
    row = dict.fromkeys(header, '')
    row.update(candidate=candidate, instance=instance,
               solved='yes' if 'Solution found!' in text else 'no')
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            row[key] = match.group(1)
    rows.append(row)
rows.sort(key=lambda r: (r['solved'] != 'yes',
          float(r['runtime']) if r['runtime'] else float('inf'), r['instance'], r['candidate']))
for name, subset in [('metrics.csv', rows)] + [
        (f'metrics-target-{f}.csv', [r for r in rows if r['instance'].startswith(f)])
        for f in ('E', 'P')]:
    with (out / 'results' / name).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(subset)
PY
echo "Listo: $OUT/results/metrics.csv"
exit "$failed"
