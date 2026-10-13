#!/usr/bin/env bash
# GBFS 5 min. Solo las 10 LLM de esta carpeta (01..10/out.rs).
# Instancias: P/dev (n16, n19-k2, n20, n22-k8). Sin nativas.
# Timeout / no solución / cargo fail no tumban el script.
# Lanzar desde WSL, en esta carpeta:  bash compile_and_run.sh

set +e
set +o pipefail
trap '' XCPU

HERE="$(cd "$(dirname "$0")" && pwd)"
PIPE="$HERE/../.."
PF="$PIPE/../mulcia-tfm-PlanForge"
DOMAIN="$PIPE/data/processed/pddl/domain.pddl"
PDDL="$PIPE/data/processed/pddl/E"
LOGS="$HERE/logs"
CSV="$HERE/metrics.csv"

INSTANCES="E-n13-k4 E-n22-k4 E-n23-k3 E-n30-k3"

mkdir -p "$LOGS"

RSS=()
for i in 01 02 03 04 05 06 07 08 09 10; do
    f="$HERE/$i/out.rs"
    if [[ ! -f "$f" ]]; then
        echo "falta $f" >&2
        exit 1
    fi
    RSS+=("$f")
done
echo "LLM .rs a ejecutar: ${#RSS[@]}"
for rs in "${RSS[@]}"; do
    echo "  - ${rs#"$HERE"/}"
done

echo "candidate,instance,solved,runtime,expanded,evaluated,generated,cost" > "$CSV"
: > "$HERE/run.log"

grab() {
    grep -oE "$2" "$1" | head -1 | grep -oE '[0-9.]+' | head -1 || true
}

run() {
    local bin="$1" h="$2" inst="$3" name="$4"
    local out="$LOGS/${name}__${inst}.txt"
    echo ">>> $name  gbfs($h())  $inst" | tee -a "$HERE/run.log"
    "$bin" --search "gbfs($h())" --max-time 300s "$DOMAIN" "$PDDL/$inst.pddl" 2>&1 | tee "$out" || true
    local solved=no
    if grep -q "Solution found!" "$out"; then
        solved=yes
    fi
    local runtime expanded evaluated generated cost
    runtime=$(grab "$out" 'Search time: [0-9.]+s')
    expanded=$(grab "$out" 'Expanded [0-9]+ state')
    evaluated=$(grab "$out" 'Evaluated [0-9]+ state')
    generated=$(grab "$out" 'Generated [0-9]+ state')
    cost=$(grab "$out" 'Plan cost: [0-9.]+')
    echo "$name,$inst,$solved,$runtime,$expanded,$evaluated,$generated,$cost" >> "$CSV"
    echo "    solved=$solved  t=$runtime  exp=$expanded  cost=$cost" | tee -a "$HERE/run.log"
}

cd "$PF"

i=0
for rs in "${RSS[@]}"; do
    i=$((i + 1))
    name=$(basename "$(dirname "$rs")")

    echo "[$i/${#RSS[@]}] copiando $name -> runner" | tee -a "$HERE/run.log"
    cp "$rs" experiments/llm-cvrp-runner/src/main.rs
    cargo build --release -p planforge-llm-cvrp 2>&1 | tee "$LOGS/compile_${name}.txt"
    compile_rc=${PIPESTATUS[0]}
    if [[ $compile_rc -ne 0 ]]; then
        echo "$name no compila" | tee -a "$HERE/run.log"
        for inst in $INSTANCES; do
            echo "$name,$inst,no,,,,,," >> "$CSV"
        done
        continue
    fi
    for inst in $INSTANCES; do
        run ./target/release/planforge-llm-cvrp llm_001 "$inst" "$name"
    done
done

echo "listo: $CSV (${#RSS[@]} LLM)" | tee -a "$HERE/run.log"
