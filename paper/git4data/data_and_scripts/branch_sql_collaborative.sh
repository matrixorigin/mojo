#!/usr/bin/env bash
set -uo pipefail

SCRIPT_NAME=$(basename "$0")
tables=(T1 T2 T3 T4)

usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME <mode> <C> <R> <G>
  mode: branch | sql
  C   : positive integer controlling the sampled row count per bucket
  R   : positive integer number of repetitions
  G   : true | false, whether rnd_keys partitions must overlap by >=10%
  
  example: ./exp2.sh branch 10000 2 false
           ./exp2.sh sql 1000 1 true
USAGE
}

if [[ $# -ne 4 ]]; then
  usage
  exit 1
fi

MODE=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
C_RAW=$2
REPEAT_RAW=$3
G_INPUT=$(printf '%s' "$4" | tr '[:upper:]' '[:lower:]')

if [[ "$MODE" != "branch" && "$MODE" != "sql" ]]; then
  echo "[error] mode must be 'branch' or 'sql'" >&2
  usage
  exit 1
fi
if ! [[ $C_RAW =~ ^[0-9]+$ ]]; then
  echo "[error] C must be a positive integer" >&2
  exit 1
fi
if ! [[ $REPEAT_RAW =~ ^[0-9]+$ ]]; then
  echo "[error] R must be a positive integer" >&2
  exit 1
fi
C=$((10#$C_RAW))
REPEAT=$((10#$REPEAT_RAW))
if (( C <= 0 )); then
  echo "[error] C must be > 0" >&2
  exit 1
fi
if (( REPEAT <= 0 )); then
  echo "[error] R must be > 0" >&2
  exit 1
fi
if [[ "$G_INPUT" != "true" && "$G_INPUT" != "false" ]]; then
  echo "[error] G must be true or false" >&2
  exit 1
fi
G_FLAG=$G_INPUT

MYSQL_BIN=${MYSQL_BIN:-mysql}
MYSQL_CMD=("$MYSQL_BIN" "-h127.0.0.1" "-P6001" "-udump" "-p111" "-D" "tpch_100g" "--batch" "--raw" "--skip-column-names" "--silent")

declare -A OP_TIMES=()
declare -a OP_ORDER=()
declare -a ERRORS=()
CURRENT_ITER=0
ROW_TARGET=$((C * 4))
if (( ROW_TARGET <= 0 )); then
  echo "[error] computed row target is invalid" >&2
  exit 1
fi
OVERLAP_COUNT=$(( (C + 9) / 10 ))
if (( OVERLAP_COUNT < 1 )); then
  OVERLAP_COUNT=1
fi
if (( OVERLAP_COUNT > C )); then
  OVERLAP_COUNT=$C
fi
CRC_THRESHOLD=$(awk -v c="$C" 'BEGIN { printf "%.8f", (c/600000000.0)*4294967296*40 }')

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

timestamp_ns() {
  local ts
  ts=$(date +%s%N 2>/dev/null) || ts=""
  if [[ $ts =~ ^[0-9]+$ ]]; then
    echo "$ts"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import time
print(int(time.time() * 1_000_000_000))
PY
    return
  fi
  if command -v perl >/dev/null 2>&1; then
    perl -MPOSIX -e 'print int(time()*1000000000), "\n";'
    return
  fi
  local seconds
  seconds=$(date +%s)
  printf '%s000000000\n' "$seconds"
}

format_duration() {
  local start_ns=$1
  local end_ns=$2
  awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.3f", (end - start) / 1000000000 }'
}

append_time() {
  local op_label=$1
  local duration=$2
  if [[ -z ${OP_TIMES[$op_label]:-} ]]; then
    OP_TIMES[$op_label]=$duration
    OP_ORDER+=("$op_label")
  else
    OP_TIMES[$op_label]+=",$duration"
  fi
}

record_error() {
  local op_label=$1
  local message=$2
  ERRORS+=("$op_label (iter ${CURRENT_ITER}): $message")
}

run_mysql() {
  local op_label=$1
  shift
  local sql_text=$1
  local start_ns end_ns status output
  start_ns=$(timestamp_ns)
  output=$("${MYSQL_CMD[@]}" -e "$sql_text" 2>&1)
  status=$?
  if (( status != 0 )); then
    record_error "$op_label" "${output//$'\n'/ } (exit $status)"
    log "MySQL error during $op_label: $output"
  elif [[ -n ${TRACE_SQL_OUTPUT:-} ]]; then
    printf '%s\n' "$output"
  fi
  end_ns=$(timestamp_ns)
  append_time "$op_label" "$(format_duration "$start_ns" "$end_ns")"
  return $status
}

safe_drop_work_tables() {
  for tbl in T0 "${tables[@]}"; do
    run_mysql "drop $tbl" "DROP TABLE IF EXISTS $tbl;" >/dev/null 2>&1 || true
  done
}

prepare_rnd_keys() {
  run_mysql "drop rnd_keys" "DROP TABLE IF EXISTS rnd_keys;" >/dev/null 2>&1 || true
  run_mysql "create rnd_keys" "CREATE TABLE rnd_keys (
  target INT NOT NULL,
  L_ORDERKEY BIGINT NOT NULL,
  L_LINENUMBER INT NOT NULL,
  PRIMARY KEY (target, L_ORDERKEY, L_LINENUMBER)
);"
  if [[ $G_FLAG == "false" ]]; then
    local sql
    sql=$(cat <<SQL
INSERT INTO rnd_keys (target, L_ORDERKEY, L_LINENUMBER)
WITH base_rows AS (
  SELECT
    L_ORDERKEY,
    L_LINENUMBER,
    ROW_NUMBER() OVER (ORDER BY L_ORDERKEY, L_LINENUMBER) AS rn
  FROM T0
  WHERE CRC32(CONCAT(L_ORDERKEY, ':', L_LINENUMBER)) < $CRC_THRESHOLD
  LIMIT $ROW_TARGET
)
SELECT
  MOD(rn - 1, 4) + 1 AS target,
  L_ORDERKEY,
  L_LINENUMBER
FROM base_rows;
SQL
)
    run_mysql "rnd_keys disjoint load" "$sql"
  else
    local sql overlap
    overlap=$OVERLAP_COUNT
    sql=$(cat <<SQL
INSERT INTO rnd_keys (target, L_ORDERKEY, L_LINENUMBER)
WITH base_rows AS (
  SELECT
    L_ORDERKEY,
    L_LINENUMBER,
    ROW_NUMBER() OVER (ORDER BY L_ORDERKEY, L_LINENUMBER) AS rn
  FROM T0
  WHERE CRC32(CONCAT(L_ORDERKEY, ':', L_LINENUMBER)) < $CRC_THRESHOLD
  LIMIT $ROW_TARGET
) 
SELECT bucket, L_ORDERKEY, L_LINENUMBER
FROM (
  SELECT
    MOD(rn - 1, 4) + 1 AS bucket,
    L_ORDERKEY,
    L_LINENUMBER,
    rn
  FROM base_rows
  UNION ALL
  SELECT
    buckets.bucket,
    br.L_ORDERKEY,
    br.L_LINENUMBER,
    br.rn
  FROM base_rows br
  JOIN (
    SELECT 1 AS bucket UNION ALL
    SELECT 2 UNION ALL
    SELECT 3 UNION ALL
    SELECT 4
  ) AS buckets
    ON buckets.bucket <> MOD(br.rn - 1, 4) + 1
  WHERE br.rn <= $overlap
) AS union_data;
SQL
)
    run_mysql "rnd_keys overlapping load" "$sql"
  fi
}

update_discount_for_table() {
  local table_name=$1
  local bucket=$2
  run_mysql "update $table_name" "UPDATE $table_name AS tgt
JOIN rnd_keys AS rk
  ON rk.target = $bucket
 AND rk.L_ORDERKEY = tgt.L_ORDERKEY
 AND rk.L_LINENUMBER = tgt.L_LINENUMBER
SET tgt.L_DISCOUNT = tgt.L_DISCOUNT + 0.01;"
}

run_branch_diffs() {
  for tbl in "${tables[@]}"; do
    run_mysql "branch diff $tbl T0" "data branch diff $tbl against T0 output count;"
  done
}

run_sql_diffs() {
  for tbl in "${tables[@]}"; do
    local sql=$(cat <<SQL
WITH unionT AS (
  SELECT -1 AS cnt, L_ORDERKEY, L_LINENUMBER, L_DISCOUNT FROM $tbl
  UNION ALL
  SELECT 1 AS cnt, L_ORDERKEY, L_LINENUMBER, L_DISCOUNT FROM T0
)
SELECT COUNT(*)
FROM (
  SELECT
    SUM(cnt) AS diffCnt,
    L_ORDERKEY,
    L_LINENUMBER,
    L_DISCOUNT
  FROM unionT
  GROUP BY L_ORDERKEY, L_LINENUMBER, L_DISCOUNT
  HAVING SUM(cnt) <> 0
) AS diff_data;
SQL
)
    run_mysql "sql diff $tbl T0" "$sql"
  done
}

run_branch_merges() {
  for tbl in "${tables[@]}"; do
    run_mysql "branch merge $tbl T0" "data branch merge $tbl into T0 when conflict accept;"
  done
}

run_sql_merge() {
  local tbl=$1
  local sql=$(cat <<SQL
START TRANSACTION;
DROP TABLE IF EXISTS diff_keys;
CREATE TABLE diff_keys (
  L_ORDERKEY   BIGINT NOT NULL,
  L_LINENUMBER INT NOT NULL,
  PRIMARY KEY (L_ORDERKEY, L_LINENUMBER)
);
INSERT INTO diff_keys (L_ORDERKEY, L_LINENUMBER)
WITH unionT AS (
  SELECT -1 AS cnt, L_ORDERKEY, L_LINENUMBER, L_DISCOUNT FROM $tbl
  UNION ALL
  SELECT  1 AS cnt, L_ORDERKEY, L_LINENUMBER, L_DISCOUNT FROM T0
)
SELECT DISTINCT L_ORDERKEY, L_LINENUMBER
FROM (
  SELECT
    SUM(cnt) AS diffCnt,
    L_ORDERKEY,
    L_LINENUMBER,
    L_DISCOUNT
  FROM unionT
  GROUP BY L_ORDERKEY, L_LINENUMBER, L_DISCOUNT
  HAVING SUM(cnt) <> 0
) AS d
WHERE d.diffCnt < 0;
DELETE FROM T0
WHERE (L_ORDERKEY, L_LINENUMBER) IN (
  SELECT L_ORDERKEY, L_LINENUMBER FROM diff_keys
);
INSERT INTO T0 (
  L_ORDERKEY, L_PARTKEY, L_SUPPKEY, L_LINENUMBER,
  L_QUANTITY, L_EXTENDEDPRICE, L_DISCOUNT, L_TAX,
  L_RETURNFLAG, L_LINESTATUS,
  L_SHIPDATE, L_COMMITDATE, L_RECEIPTDATE,
  L_SHIPINSTRUCT, L_SHIPMODE, L_COMMENT
)
SELECT
  L_ORDERKEY, L_PARTKEY, L_SUPPKEY, L_LINENUMBER,
  L_QUANTITY, L_EXTENDEDPRICE, L_DISCOUNT, L_TAX,
  L_RETURNFLAG, L_LINESTATUS,
  L_SHIPDATE, L_COMMITDATE, L_RECEIPTDATE,
  L_SHIPINSTRUCT, L_SHIPMODE, L_COMMENT
FROM $tbl
WHERE (L_ORDERKEY, L_LINENUMBER) IN (
  SELECT L_ORDERKEY, L_LINENUMBER FROM diff_keys
);
DROP TABLE diff_keys;
COMMIT;
SQL
)
  run_mysql "sql merge $tbl T0" "$sql"
}

safe_drop_work_tables

for (( run=1; run<=REPEAT; run++ )); do
  CURRENT_ITER=$run
  log "Run $run/$REPEAT - resetting tables"
  safe_drop_work_tables
  run_mysql "create T0" "CREATE TABLE T0 CLONE lineitem{snapshot=\"sp_base\"};"
  if [[ $MODE == "branch" ]]; then
    for tbl in "${tables[@]}"; do
      run_mysql "branch clone $tbl" "CREATE TABLE $tbl CLONE T0;"
    done
  else
    for tbl in "${tables[@]}"; do
     run_mysql "sql copy $tbl" "CREATE TABLE $tbl SELECT * FROM T0;"
    done
  fi

  log "Run $run/$REPEAT - preparing rnd_keys"
  prepare_rnd_keys

  log "Run $run/$REPEAT - updating discount"
  bucket=1
  for tbl in "${tables[@]}"; do
    update_discount_for_table "$tbl" $bucket
    ((bucket++))
  done
  run_mysql "drop rnd_keys" "DROP TABLE IF EXISTS rnd_keys;"

  log "Run $run/$REPEAT - diff stage"
  if [[ $MODE == "branch" ]]; then
    run_branch_diffs
  else
    run_sql_diffs
  fi

  log "Run $run/$REPEAT - merge stage"
  if [[ $MODE == "branch" ]]; then
    run_branch_merges
  else
    for tbl in "${tables[@]}"; do
      run_sql_merge "$tbl"
    done
  fi
done

echo ""
echo "Timing summary:"
for op in "${OP_ORDER[@]}"; do
  printf '%s: %s\n' "$op" "${OP_TIMES[$op]}"
done

if (( ${#ERRORS[@]} > 0 )); then
  echo ""
  echo "Errors captured:"
  for err in "${ERRORS[@]}"; do
    echo "- $err"
  done
else
  echo ""
  echo "No errors captured."
fi
