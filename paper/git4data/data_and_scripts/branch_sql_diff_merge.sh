#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: exp1.sh <create_mode> <C> <runs>
  create_mode: branch | sql
  C          : positive integer count used in sampling
  runs       : positive integer number of experiment repetitions
  
  example: ./exp1.sh branch 1000 5
           ./exp1.sh sql 1000 1

Environment (override with export before running):
  MYSQL_BIN       (default: mysql)
  MYSQL_HOST      (default: 127.0.0.1)
  MYSQL_PORT      (default: 6001)
  MYSQL_USER      (default: dump)
  MYSQL_PASSWORD  (default: 111, unset for passwordless)
  MYSQL_DB        (default: tpch_100g)
EOF
  exit 1
}

if [[ $# -ne 3 ]]; then
  usage
fi

CREATE_MODE=$1
C_RAW=$2
RUNS_RAW=$3

if [[ "$CREATE_MODE" != "branch" && "$CREATE_MODE" != "sql" ]]; then
  echo "Invalid create_mode: $CREATE_MODE (expected branch or sql)" >&2
  usage
fi

if ! [[ "$C_RAW" =~ ^[0-9]+$ ]]; then
  echo "C must be a positive integer" >&2
  usage
fi

C=$((10#$C_RAW))
if (( C <= 0 )); then
  echo "C must be greater than 0" >&2
  usage
fi

if ! [[ "$RUNS_RAW" =~ ^[0-9]+$ ]]; then
  echo "runs must be a positive integer" >&2
  usage
fi

RUNS=$((10#$RUNS_RAW))
if (( RUNS <= 0 )); then
  echo "runs must be greater than 0" >&2
  usage
fi

MYSQL_BIN=${MYSQL_BIN:-mysql}
MYSQL_HOST=${MYSQL_HOST:-127.0.0.1}
MYSQL_PORT=${MYSQL_PORT:-6001}
MYSQL_USER=${MYSQL_USER:-dump}
MYSQL_PASSWORD=${MYSQL_PASSWORD-111}
MYSQL_DB=${MYSQL_DB:-tpch_100g}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for timing measurements" >&2
  exit 1
fi

THRESHOLD=$(python3 -c 'import sys; C=int(sys.argv[1]); print((C/600000000.0)*(2**32)*10)' "$C")

declare -a MYSQL_ARGS
MYSQL_ARGS=(-h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER")
if [[ -n "$MYSQL_DB" ]]; then
  MYSQL_ARGS+=(-D "$MYSQL_DB")
fi

run_sql() {
  local sql="$1"
  if [[ -n "$MYSQL_PASSWORD" ]]; then
    MYSQL_PWD="$MYSQL_PASSWORD" "$MYSQL_BIN" "${MYSQL_ARGS[@]}" -e "$sql"
  else
    "$MYSQL_BIN" "${MYSQL_ARGS[@]}" -e "$sql"
  fi
}

declare -a TIMING_LABELS=()
declare -a TIMING_VALUES=()

measure() {
  local label=$1
  shift
  local start end duration status
  start=$(python3 -c 'import time; print(time.time())')
  set +e
  "$@"
  status=$?
  set -e
  end=$(python3 -c 'import time; print(time.time())')
  if [[ $status -ne 0 ]]; then
    echo "ERROR: $label failed (exit $status)" >&2
    return $status
  fi
  duration=$(python3 -c 'import sys; print(f"{float(sys.argv[2])-float(sys.argv[1]):.3f}")' "$start" "$end")
  local found_index=-1
  for idx in "${!TIMING_LABELS[@]}"; do
    if [[ "${TIMING_LABELS[idx]}" == "$label" ]]; then
      found_index=$idx
      break
    fi
  done

  if (( found_index == -1 )); then
    TIMING_LABELS+=("$label")
    TIMING_VALUES+=("$duration")
  else
    TIMING_VALUES[$found_index]="${TIMING_VALUES[$found_index]},$duration"
  fi
}

cleanup_needed=0
cleanup() {
  if [[ $cleanup_needed -eq 1 ]]; then
    set +e
    run_sql 'DROP TABLE IF EXISTS rnd_keys;'
    set -e
  fi
}
trap cleanup EXIT

for ((run=1; run<=RUNS; run++)); do
  echo "===== Run ${run}/${RUNS} ====="
  echo "Step 0: prepare T1/T2 using mode=${CREATE_MODE}"
  run_sql 'DROP TABLE IF EXISTS T1;'
  if [[ "$CREATE_MODE" == "branch" ]]; then
    measure "create T1 (branch clone)" run_sql 'CREATE TABLE T1 CLONE T0{snapshot="sp_base"};'
  else
    measure "create T1 (sql copy)" run_sql 'CREATE TABLE T1 AS SELECT * FROM T0{snapshot="sp_base"};'
  fi

  run_sql 'DROP TABLE IF EXISTS T2;'
  if [[ "$CREATE_MODE" == "branch" ]]; then
    measure "create T2 (branch clone)" run_sql 'CREATE TABLE T2 CLONE T0{snapshot="sp_base"};'
  else
    measure "create T2 (sql copy)" run_sql 'CREATE TABLE T2 AS SELECT * FROM T0{snapshot="sp_base"};'
  fi

  echo "Step 1: update T2 (C=${C})"
  run_sql 'DROP TABLE IF EXISTS rnd_keys;'
  run_sql $'CREATE TABLE rnd_keys (\n  l_orderkey BIGINT,\n  l_linenumber INT,\n  PRIMARY KEY (l_orderkey, l_linenumber)\n);'
  cleanup_needed=1

#INSERT INTO rnd_keys
#SELECT l_orderkey, l_linenumber
#FROM T2
#WHERE CRC32(CONCAT(l_orderkey, ':', l_linenumber)) < ${THRESHOLD}
#LIMIT ${C};


  INSERT_RND_KEYS=$(cat <<SQL

INSERT INTO rnd_keys
SELECT l_orderkey, l_linenumber
FROM T2
WHERE CRC32(CONCAT(l_orderkey, ':', l_linenumber)) < ${THRESHOLD}
LIMIT ${C};

SQL
)
  measure "populate rnd_keys" run_sql "$INSERT_RND_KEYS"

  UPDATE_T2=$(cat <<SQL
UPDATE T2
JOIN (
  SELECT l_orderkey, l_linenumber
  FROM rnd_keys
  ORDER BY l_orderkey, l_linenumber
  LIMIT ${C}
) b
ON T2.l_orderkey = b.l_orderkey AND T2.l_linenumber = b.l_linenumber
SET T2.l_discount = T2.l_discount + 0.01;
SQL
)
  measure "update T2" run_sql "$UPDATE_T2"

  run_sql 'DROP TABLE IF EXISTS rnd_keys;'
  cleanup_needed=0

  echo "Step 2: diff/merge"
  if [[ "$CREATE_MODE" == "branch" ]]; then
    measure "branch diff T2 vs T1" run_sql 'DATA BRANCH DIFF T2 AGAINST T1 OUTPUT COUNT;'
    measure "branch merge T2 into T1" run_sql 'DATA BRANCH MERGE T2 INTO T1;'
  else
    SQL_DIFF=$(cat <<'SQL'
WITH unionT AS (
  SELECT -1 AS cnt, L_ORDERKEY, L_LINENUMBER, L_DISCOUNT FROM T2
  UNION ALL
  SELECT 1 AS cnt, L_ORDERKEY, L_LINENUMBER, L_DISCOUNT FROM T1
)
select count(*) from (
  SELECT
    SUM(cnt) AS diffCnt,
    L_ORDERKEY,
    L_LINENUMBER,
    L_DISCOUNT
  FROM unionT
  GROUP BY L_ORDERKEY, L_LINENUMBER, L_DISCOUNT
  HAVING SUM(cnt) <> 0
);
SQL
)
    measure "sql diff T2 vs T1" run_sql "$SQL_DIFF"

    SQL_MERGE=$(cat <<'SQL'


START TRANSACTION;

CREATE TABLE diff_keys (
  L_ORDERKEY   BIGINT NOT NULL,
  L_LINENUMBER INT    NOT NULL,
  PRIMARY KEY (L_ORDERKEY, L_LINENUMBER)
) ;

INSERT INTO diff_keys (L_ORDERKEY, L_LINENUMBER)
WITH unionT AS (
  SELECT -1 AS cnt, L_ORDERKEY, L_LINENUMBER, L_DISCOUNT FROM T2
  UNION ALL
  SELECT  1 AS cnt, L_ORDERKEY, L_LINENUMBER, L_DISCOUNT FROM T1
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

DELETE FROM T1
WHERE (L_ORDERKEY, L_LINENUMBER) IN (
  SELECT L_ORDERKEY, L_LINENUMBER FROM diff_keys
);

INSERT INTO T1 (
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
FROM T2
WHERE (L_ORDERKEY, L_LINENUMBER) IN (
  SELECT L_ORDERKEY, L_LINENUMBER FROM diff_keys
);

COMMIT;

DROP TABLE IF EXISTS diff_keys;


SQL
)
    measure "sql merge T2 into T1" run_sql "$SQL_MERGE"
  fi
done

if (( ${#TIMING_LABELS[@]} > 0 )); then
  echo "Timings (per run):"
  for idx in "${!TIMING_LABELS[@]}"; do
    IFS=',' read -r -a per_run <<< "${TIMING_VALUES[idx]}"
    printf "  %s: " "${TIMING_LABELS[idx]}"
    for dur_idx in "${!per_run[@]}"; do
      printf "%s%ss" \
        "$( [[ $dur_idx -gt 0 ]] && printf ',' )" \
        "${per_run[dur_idx]}"
    done
    printf "\n"
  done
fi
