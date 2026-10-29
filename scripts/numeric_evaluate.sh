#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -eq 5 ]] || {
    echo "Usage: $0 FAMILY CANDIDATES_DIR OUTPUT_DIR ARTIFACTS_DIR PLANFORGE_DIR" >&2
    exit 2
}

for required_command in python3 cargo flock timeout mktemp; do
    command -v "$required_command" >/dev/null 2>&1 || {
        echo "Missing required command inside WSL: $required_command" >&2
        exit 127
    }
done

FAMILY="$1"
CANDIDATES="$(cd "$2" && pwd)"
OUT="$3"
ARTIFACTS="$(cd "$4" && pwd)"
PLANFORGE="$(cd "$5" && pwd)"
[[ "$FAMILY" == E || "$FAMILY" == P ]] || { echo "Family must be E or P" >&2; exit 2; }

DOMAIN="$ARTIFACTS/inputs/02-numeric-planning/02-pddl-converted/domain.pddl"
PDDL="$ARTIFACTS/inputs/02-numeric-planning/02-pddl-converted/$FAMILY"
SPLIT="$ARTIFACTS/inputs/02-numeric-planning/03-splits/$FAMILY/dev.json"
MAIN="$PLANFORGE/experiments/llm-cvrp-runner/src/main.rs"
[[ -f "$DOMAIN" && -f "$SPLIT" && -f "$MAIN" ]] || { echo "Missing input or PlanForge runner" >&2; exit 1; }
[[ ! -e "$OUT/metrics.csv" && ! -e "$OUT/logs" ]] || {
    echo "Output already contains an evaluation: $OUT" >&2
    exit 1
}

mkdir -p "$OUT/logs" "$OUT/plans"
OUT="$(cd "$OUT" && pwd)"
mapfile -t INSTANCES < <(python3 - "$SPLIT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(*json.load(stream)["instances"], sep="\n")
PY
)

WORK="$(mktemp -d)"
cp -p "$MAIN" "$WORK/main.rs.before"
cleanup() {
    local rc=$?
    cp -p "$WORK/main.rs.before" "$MAIN"
    rm -rf "$WORK"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>"$PLANFORGE/.heuristic-evaluation.lock"
flock -n 9 || { echo "Another PlanForge evaluation is active" >&2; exit 1; }
export CARGO_TARGET_DIR="$PLANFORGE/target"
export LC_ALL=C

echo 'candidate,instance,solved,runtime,expanded,evaluated,generated,cost' > "$OUT/metrics.csv"

grab() {
    grep -oE "$2" "$1" | head -1 | grep -oE '[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?' | head -1 || true
}

for number in 01 02 03 04 05 06 07 08 09 10; do
    source="$CANDIDATES/$number/out.rs"
    [[ -f "$source" ]] || { echo "Missing candidate: $source" >&2; exit 1; }
    cp "$source" "$MAIN"
    build_log="$OUT/logs/build-$number.log"
    build_rc=0
    (cd "$PLANFORGE" && cargo build --release --locked -p planforge-llm-cvrp) \
        > "$build_log" 2>&1 || build_rc=$?
    echo "Build exit: $build_rc" >> "$build_log"
    if [[ $build_rc -ne 0 ]]; then
        for instance in "${INSTANCES[@]}"; do
            echo "$number,$instance,no,,,,," >> "$OUT/metrics.csv"
        done
        continue
    fi

    for instance in "${INSTANCES[@]}"; do
        name="${number}__${instance}"
        log="$OUT/logs/$name.log"
        trial="$WORK/$name"
        mkdir "$trial"
        rc=0
        (cd "$trial" && timeout --kill-after=5s 330s \
            "$PLANFORGE/target/release/planforge-llm-cvrp" \
            --search 'gbfs(llm_001())' --max-time 300s --max-memory 4G \
            "$DOMAIN" "$PDDL/$instance.pddl") > "$log" 2>&1 || rc=$?
        echo "Trial exit: $rc" >> "$log"
        solved=no
        grep -q 'Solution found!' "$log" && solved=yes
        [[ -f "$trial/sas_plan" ]] && cp "$trial/sas_plan" "$OUT/plans/$name.sas_plan"
        runtime="$(grab "$log" 'Search time: [0-9.eE+-]+s')"
        expanded="$(grab "$log" 'Expanded [0-9]+ state')"
        evaluated="$(grab "$log" 'Evaluated [0-9]+ state')"
        generated="$(grab "$log" 'Generated [0-9]+ state')"
        cost="$(grab "$log" 'Plan cost: [0-9.eE+-]+')"
        echo "$number,$instance,$solved,$runtime,$expanded,$evaluated,$generated,$cost" >> "$OUT/metrics.csv"
    done
done

echo "Evaluation written to $OUT/metrics.csv"
