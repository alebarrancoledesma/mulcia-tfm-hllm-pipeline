#!/usr/bin/env bash
# Compila y corre las heurísticas LLM + built-ins de PlanForge (GBFS, 300s).
# Solo bash + cargo: copia cada .rs al runner, compila y ejecuta.
#
#   bash run_gbfs_300s.sh
#
# Override: PLANFORGE_ROOT=/ruta/mulcia-tfm-PlanForge
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PLANFORGE_ROOT="${PLANFORGE_ROOT:-${PIPELINE_ROOT}/../mulcia-tfm-PlanForge}"

DOMAIN="${PIPELINE_ROOT}/data/processed/pddl/domain.pddl"
PDDL_P="${PIPELINE_ROOT}/data/processed/pddl/P"
RUNNER_DIR="${PLANFORGE_ROOT}/experiments/llm-cvrp-runner"
RUNNER_MAIN="${RUNNER_DIR}/src/main.rs"
BIN_PLANFORGE="${PLANFORGE_ROOT}/target/release/planforge"
BIN_LLM="${PLANFORGE_ROOT}/target/release/planforge-llm-cvrp"

CSV="${SCRIPT_DIR}/gbfs_300s_metrics.csv"
LOG="${SCRIPT_DIR}/gbfs_300s_run.log"
COMPILE_LOG_DIR="${SCRIPT_DIR}/compile_logs"

# n48/n65 + subir --max-ground-atoms tumba WSL (OOM ~10GB; systemd mata Ubuntu).
# Tres tamaños que caben en el default de 10M átomos, ninguna en el prompt.
INSTANCES=(P-n16-k8 P-n20-k2 P-n23-k8)
BUILTINS=(blind ff lmcutnumeric)
SEARCH=gbfs
MAX_TIME_S=300
MAX_MEMORY=2G
MAX_GROUNDING_MEMORY=2G
SUBPROCESS_S=900

csv_escape() {
  local s="${1//$'\n'/ }"
  s="${s//$'\r'/ }"
  s="${s//\"/\"\"}"
  printf '"%s"' "${s}"
}

append_row() {
  local candidate="$1" instance="$2" solved="$3" runtime="$4"
  local expanded="$5" evaluated="$6" generated="$7" cost="$8"
  local timeout="$9" error="${10:-}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(csv_escape "${candidate}")" \
    "$(csv_escape "${instance}")" \
    "$(csv_escape "${SEARCH}")" \
    "${solved}" "${runtime}" "${expanded}" "${evaluated}" "${generated}" "${cost}" \
    "${timeout}" \
    "$(csv_escape "${error}")" >> "${CSV}"
}

parse_field() {
  local file="$1" regex="$2"
  grep -oE "${regex}" "${file}" | head -n 1 | grep -oE '[0-9.]+' | head -n 1
}

run_one() {
  local binary="$1" heuristic="$2" instance="$3" candidate_label="$4"
  local pddl="${PDDL_P}/${instance}.pddl"
  local runlog="${COMPILE_LOG_DIR}/run__${candidate_label//[^A-Za-z0-9._-]/_}__${instance}.log"

  local args=(
    "${binary}"
    --search "${SEARCH}(${heuristic}())"
    --max-time "${MAX_TIME_S}s"
    --max-memory "${MAX_MEMORY}"
    --max-grounding-memory "${MAX_GROUNDING_MEMORY}"
    "${DOMAIN}" "${pddl}"
  )

  echo ">>> ${candidate_label}  ${SEARCH}(${heuristic}())  ${instance}" | tee -a "${LOG}"

  local rc
  timeout "${SUBPROCESS_S}s" "${args[@]}" > "${runlog}" 2>&1
  rc=$?

  local solved=False timeout_flag=False runtime="" expanded="" evaluated="" generated="" cost="" error=""

  if [[ ${rc} -eq 124 ]]; then
    timeout_flag=True
    error="timeout de subprocess (el planner no se autoterminó)"
    append_row "${candidate_label}" "${instance}" "${solved}" "${runtime}" \
      "${expanded}" "${evaluated}" "${generated}" "${cost}" "${timeout_flag}" "${error}"
    echo "    timeout subprocess rc=${rc}" | tee -a "${LOG}"
    return 0
  fi
  if [[ ${rc} -eq 7 ]]; then
    timeout_flag=True
    runtime="$(parse_field "${runlog}" 'Search time: [0-9.]+s')"
    append_row "${candidate_label}" "${instance}" "${solved}" "${runtime}" \
      "${expanded}" "${evaluated}" "${generated}" "${cost}" "${timeout_flag}" "${error}"
    echo "    planner timeout rc=7" | tee -a "${LOG}"
    return 0
  fi
  if [[ ${rc} -eq 6 ]]; then
    error="out of memory"
    runtime="$(parse_field "${runlog}" 'Search time: [0-9.]+s')"
    append_row "${candidate_label}" "${instance}" "${solved}" "${runtime}" \
      "${expanded}" "${evaluated}" "${generated}" "${cost}" "${timeout_flag}" "${error}"
    echo "    OOM rc=6" | tee -a "${LOG}"
    return 0
  fi
  if [[ ${rc} -ne 0 ]]; then
    error="exit code ${rc}: $(tail -c 1500 "${runlog}")"
    append_row "${candidate_label}" "${instance}" "${solved}" "${runtime}" \
      "${expanded}" "${evaluated}" "${generated}" "${cost}" "${timeout_flag}" "${error}"
    echo "    FALLO rc=${rc}" | tee -a "${LOG}"
    tail -n 40 "${runlog}" | tee -a "${LOG}"
    return 0
  fi

  expanded="$(parse_field "${runlog}" 'Expanded [0-9]+ state')"
  evaluated="$(parse_field "${runlog}" 'Evaluated [0-9]+ state')"
  generated="$(parse_field "${runlog}" 'Generated [0-9]+ state')"
  cost="$(parse_field "${runlog}" 'Plan cost: [0-9.]+')"
  runtime="$(parse_field "${runlog}" 'Search time: [0-9.]+s')"

  if grep -q 'Solution found!' "${runlog}"; then
    solved=True
  elif grep -qE 'No solution found|Search timed out' "${runlog}"; then
    solved=False
  else
    error="no se pudo parsear la salida de PlanForge"
  fi

  append_row "${candidate_label}" "${instance}" "${solved}" "${runtime}" \
    "${expanded}" "${evaluated}" "${generated}" "${cost}" "${timeout_flag}" "${error}"
  echo "    solved=${solved} runtime=${runtime} expanded=${expanded} cost=${cost}" | tee -a "${LOG}"
}

# --- preflight ---
command -v cargo >/dev/null || { echo "ERROR: cargo no está en el PATH"; exit 1; }
command -v timeout >/dev/null || { echo "ERROR: timeout (coreutils) no está en el PATH"; exit 1; }
[[ -d "${PLANFORGE_ROOT}" ]] || { echo "ERROR: no existe PlanForge en ${PLANFORGE_ROOT}"; exit 1; }
[[ -f "${DOMAIN}" ]] || { echo "ERROR: no existe ${DOMAIN}"; exit 1; }
[[ -f "${RUNNER_MAIN}" ]] || { echo "ERROR: no existe ${RUNNER_MAIN}"; exit 1; }
[[ ! -f "${CSV}" ]] || { echo "ERROR: ${CSV} ya existe. Renómbralo o bórralo."; exit 1; }

mkdir -p "${COMPILE_LOG_DIR}"
printf '%s\n' "candidate,instance,search,solved,runtime,expanded,evaluated,generated,solution_cost,timeout,error" > "${CSV}"
: > "${LOG}"

echo "PlanForge: ${PLANFORGE_ROOT}" | tee -a "${LOG}"
echo "Instancias: ${INSTANCES[*]}" | tee -a "${LOG}"
echo "Límites: --max-memory ${MAX_MEMORY}  --max-grounding-memory ${MAX_GROUNDING_MEMORY}" | tee -a "${LOG}"
echo "CSV: ${CSV}" | tee -a "${LOG}"
free -h | tee -a "${LOG}"
echo | tee -a "${LOG}"

# --- 1) binario base + built-ins ---
cd "${PLANFORGE_ROOT}"

echo "Compilando planforge..." | tee -a "${LOG}"
cargo build --release -p planforge > "${COMPILE_LOG_DIR}/planforge.log" 2>&1
rc=$?
if [[ ${rc} -ne 0 || ! -x "${BIN_PLANFORGE}" ]]; then
  echo "NO COMPILA planforge" | tee -a "${LOG}"
  tail -n 40 "${COMPILE_LOG_DIR}/planforge.log" | tee -a "${LOG}"
  for h in "${BUILTINS[@]}"; do
    for inst in "${INSTANCES[@]}"; do
      append_row "${h}" "${inst}" False "" "" "" "" "" False "compile_error: planforge base"
    done
  done
else
  echo "planforge compilado." | tee -a "${LOG}"
  for h in "${BUILTINS[@]}"; do
    for inst in "${INSTANCES[@]}"; do
      run_one "${BIN_PLANFORGE}" "${h}" "${inst}" "${h}"
    done
  done
fi

# --- 2) cada .rs LLM: copiar -> compilar runner -> ejecutar ---
shopt -s nullglob
mapfile -t RS_FILES < <(find "${SCRIPT_DIR}" -mindepth 2 -maxdepth 2 -type f -name 'out - *.rs' | sort)
shopt -u nullglob

if [[ ${#RS_FILES[@]} -eq 0 ]]; then
  echo "ERROR: no hay out - *.rs en ${SCRIPT_DIR}/*/" | tee -a "${LOG}"
  exit 1
fi

for rs in "${RS_FILES[@]}"; do
  prompt_dir="$(basename "$(dirname "${rs}")")"
  stem="$(basename "${rs}" .rs)"
  label="${prompt_dir} / ${stem}"
  clog="${COMPILE_LOG_DIR}/${prompt_dir}__${stem}.log"

  echo "Copiando ${label} -> ${RUNNER_MAIN}" | tee -a "${LOG}"
  cp "${rs}" "${RUNNER_MAIN}"

  echo "Compilando ${label}..." | tee -a "${LOG}"
  cargo build --release -p planforge-llm-cvrp > "${clog}" 2>&1
  rc=$?

  if [[ ${rc} -ne 0 || ! -x "${BIN_LLM}" ]]; then
    echo "${label}: NO COMPILA" | tee -a "${LOG}"
    tail -n 40 "${clog}" | tee -a "${LOG}"
    err="compile_error: $(tail -c 1000 "${clog}" | tr '\n' ' ')"
    for inst in "${INSTANCES[@]}"; do
      append_row "${label}" "${inst}" False "" "" "" "" "" False "${err}"
    done
    continue
  fi

  echo "${label}: compilado." | tee -a "${LOG}"
  for inst in "${INSTANCES[@]}"; do
    run_one "${BIN_LLM}" llm_001 "${inst}" "${label}"
  done
done

echo | tee -a "${LOG}"
echo "Listo. CSV: ${CSV}" | tee -a "${LOG}"
