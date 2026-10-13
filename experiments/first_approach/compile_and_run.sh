#!/usr/bin/env bash
# GBFS 5 min, P-n16-k8 + P-n20-k2 + P-n23-k8.
# Builtins (blind/ff/lmcutnumeric) reutilizados de first_approach_01.
# Las 8 heurísticas LLM de esta carpeta se lanzan.
# Timeout / no solución / cargo fail no tumban el script.
# Lanzar desde WSL, en esta carpeta:  bash compile_and_run.sh

HERE="$(cd "$(dirname "$0")" && pwd)"
PIPE="$HERE/../.."
PF="$PIPE/../mulcia-tfm-PlanForge"
DOMAIN="$PIPE/data/processed/pddl/domain.pddl"
PDDL="$PIPE/data/processed/pddl/P"
LOGS="$HERE/logs"
CSV="$HERE/metrics.csv"

INSTANCES="P-n16-k8 P-n20-k2 P-n23-k8"

mkdir -p "$LOGS"

RSS=()
for prompt in main_prompt main_prompt_enhanced_00 main_prompt_enhanced_01 main_prompt_enhanced_02; do
    for who in "openai - gpt-5.6-sol - medium" "deepseek - deepseek-v4-pro - medium"; do
        f="$HERE/$prompt/out - $who.rs"
        if [[ ! -f "$f" ]]; then
            echo "falta $f" >&2
            exit 1
        fi
        RSS+=("$f")
    done
done
echo "LLM .rs a ejecutar: ${#RSS[@]}"
for rs in "${RSS[@]}"; do
    echo "  - ${rs#"$HERE"/}"
done

# builtins: copiados de first_approach_01, no se relanzan
echo "candidate,instance,solved,runtime,expanded,evaluated,generated,cost" > "$CSV"
cat >> "$CSV" <<'EOF'
blind,P-n16-k8,yes,13.802445,5224907,5225076,38244603,468.000000
blind,P-n20-k2,no,300.000010,41007840,113760817,486355824,
blind,P-n23-k8,no,300.000071,26623782,110638090,603006127,
ff,P-n16-k8,yes,0.418045,81232,73985,430893,540.000000
ff,P-n20-k2,yes,6.632133,1407582,252931,5748928,529.000000
ff,P-n23-k8,no,,,,,
lmcutnumeric,P-n16-k8,no,,,,,
lmcutnumeric,P-n20-k2,no,,,,,
lmcutnumeric,P-n23-k8,no,,,,,
EOF

{
    echo "builtins reutilizados de first_approach_01 (no se relanzan)"
    echo ">>> blind  gbfs(blind())  P-n16-k8"
    echo "    solved=yes  t=13.802445  exp=5224907  cost=468.000000"
    echo ">>> blind  gbfs(blind())  P-n20-k2"
    echo "    solved=no  t=300.000010  exp=41007840  cost="
    echo ">>> blind  gbfs(blind())  P-n23-k8"
    echo "    solved=no  t=300.000071  exp=26623782  cost="
    echo ">>> ff  gbfs(ff())  P-n16-k8"
    echo "    solved=yes  t=0.418045  exp=81232  cost=540.000000"
    echo ">>> ff  gbfs(ff())  P-n20-k2"
    echo "    solved=yes  t=6.632133  exp=1407582  cost=529.000000"
    echo ">>> ff  gbfs(ff())  P-n23-k8"
    echo "    solved=no  t=  exp=  cost="
    echo ">>> lmcutnumeric  gbfs(lmcutnumeric())  P-n16-k8"
    echo "    solved=no  t=  exp=  cost="
    echo ">>> lmcutnumeric  gbfs(lmcutnumeric())  P-n20-k2"
    echo "    solved=no  t=  exp=  cost="
    echo ">>> lmcutnumeric  gbfs(lmcutnumeric())  P-n23-k8"
    echo "    solved=no  t=  exp=  cost="
    echo "lanzando ${#RSS[@]} heurísticas LLM"
} > "$HERE/run.log"

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
    prompt=$(basename "$(dirname "$rs")")
    stem=$(basename "$rs" .rs)
    who=$(echo "$stem" | sed 's/^out - //; s/ - medium$//; s/ - /_/g')
    name="${prompt}__${who}"

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

echo "listo: $CSV (${#RSS[@]} LLM + builtins)" | tee -a "$HERE/run.log"
