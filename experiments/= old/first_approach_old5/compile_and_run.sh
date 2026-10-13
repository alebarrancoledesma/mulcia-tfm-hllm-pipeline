#!/usr/bin/env bash
# GBFS 5 min, P-n16-k8 + P-n20-k2 + P-n23-k8. Builtins + LLM .rs.
# Lanzar desde WSL, en esta carpeta:  bash run_one.sh

HERE="$(cd "$(dirname "$0")" && pwd)"
PIPE="$HERE/../.."
PF="$PIPE/../mulcia-tfm-PlanForge"
DOMAIN="$PIPE/data/processed/pddl/domain.pddl"
PDDL="$PIPE/data/processed/pddl/P"
LOGS="$HERE/logs"
CSV="$HERE/metrics.csv"

INSTANCES="P-n16-k8 P-n20-k2 P-n23-k8"
BUILTINS="blind ff lmcutnumeric"

mkdir -p "$LOGS"
: > "$HERE/run.log"
echo "candidate,instance,solved,runtime,expanded,evaluated,generated,cost" > "$CSV"

grab() {
    grep -oE "$2" "$1" | head -1 | grep -oE '[0-9.]+' | head -1
}

run() {
    local bin="$1" h="$2" inst="$3" name="$4"
    local out="$LOGS/${name}__${inst}.txt"
    echo ">>> $name  gbfs($h())  $inst" | tee -a "$HERE/run.log"
    "$bin" --search "gbfs($h())" --max-time 300s "$DOMAIN" "$PDDL/$inst.pddl" 2>&1 | tee "$out"
    local solved=no
    grep -q "Solution found!" "$out" && solved=yes
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

echo "compilando planforge..." | tee -a "$HERE/run.log"
cargo build --release -p planforge 2>&1 | tee "$LOGS/compile_planforge.txt"
for h in $BUILTINS; do
    for inst in $INSTANCES; do
        run ./target/release/planforge "$h" "$inst" "$h"
    done
done

for rs in "$HERE"/*/out\ -\ *.rs; do
    prompt=$(basename "$(dirname "$rs")")
    stem=$(basename "$rs" .rs)
    # "out - openai - gpt-5.6-sol - medium" -> openai_gpt-5.6-sol
    who=$(echo "$stem" | sed 's/^out - //; s/ - medium$//; s/ - /_/g')
    name="${prompt}__${who}"

    echo "copiando $name -> runner" | tee -a "$HERE/run.log"
    cp "$rs" experiments/llm-cvrp-runner/src/main.rs
    cargo build --release -p planforge-llm-cvrp 2>&1 | tee "$LOGS/compile_${name}.txt"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
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

echo "listo: $CSV" | tee -a "$HERE/run.log"
