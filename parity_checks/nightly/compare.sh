#!/usr/bin/env bash
# Nightly parity comparison for one suite (GH #702). Called by
# .github/workflows/nightly-parity.yml once a suite's runner has written its report.
#
#   compare.sh SUITE KIND REPORT[,REPORT...] [extra `verdicts.py diff` args, e.g. --expect-backend cc]
#
#   1. extract REPORT (several comma-separated reports -- passes of one sweep -- are
#      merged) into $NIGHTLY_OUT/SUITE/verdicts.json. That file is the candidate
#      baseline: committing it as baselines/SUITE.json accepts tonight's state.
#   2. diff it against the committed baseline, parity_checks/nightly/baselines/SUITE.json.
#   3. diff it against the previous nightly's verdicts for SUITE (the last scheduled or
#      manual run on main), when one can be downloaded. This catches a fix that
#      regresses before anyone re-baselined it.
#
# Both Markdown reports go to $GITHUB_STEP_SUMMARY. Exit 1 on any alert or on a
# missing report; a missing baseline is an alert unless ALLOW_MISSING_BASELINE is set
# (only while a baseline is being bootstrapped on a pull request).
set -uo pipefail

suite=$1
kind=$2
IFS=',' read -r -a reports <<<"$3"
shift 3

here=$(cd "$(dirname "$0")" && pwd)
out=${NIGHTLY_OUT:-nightly-out}/$suite
summary=${GITHUB_STEP_SUMMARY:-/dev/null}
mkdir -p "$out"

extract_args=()
i=0
for report in "${reports[@]}"; do
  if [ ! -s "$report" ]; then
    echo "::error::$suite: no report at $report; the run did not finish."
    printf '### %s\n\n**No report** (%s): the run did not finish. See the job log.\n\n' \
      "$suite" "$(basename "$report")" >>"$summary"
    printf '{"suite": "%s", "label": "", "no_report": true, "n_alerts": 1}\n' "$suite" >"$out/diff-baseline.json"
    exit 1
  fi
  if [ "$i" -eq 0 ]; then cp "$report" "$out/report.json"; else cp "$report" "$out/report-$i.json"; fi
  extract_args+=(--report "$report")
  i=$((i + 1))
done

prov=$(printf '{"run_id": "%s", "sha": "%s", "event": "%s"}' \
  "${GITHUB_RUN_ID:-local}" "${GITHUB_SHA:-}" "${GITHUB_EVENT_NAME:-local}")
if ! python3 "$here/verdicts.py" extract --kind "$kind" --suite "$suite" \
  "${extract_args[@]}" --out "$out/verdicts.json" --provenance "$prov"; then
  echo "::error::$suite: could not extract verdicts from ${reports[*]}"
  exit 1
fi

rc=0
allow=()
if [ -n "${ALLOW_MISSING_BASELINE:-}" ]; then
  allow=(--allow-missing-baseline)
fi
python3 "$here/verdicts.py" diff \
  --baseline "$here/baselines/$suite.json" --fresh "$out/verdicts.json" \
  --label "vs committed baseline" --flaky "$here/flaky.json" \
  --json-out "$out/diff-baseline.json" --md-out "$out/diff-baseline.md" \
  ${allow[@]+"${allow[@]}"} "$@" >/dev/null || rc=1
if [ -f "$out/diff-baseline.md" ]; then
  cat "$out/diff-baseline.md" >>"$summary"
else
  echo "::error::$suite: the baseline comparison failed to run"
  rc=1
fi

# The previous nightly's verdicts for this suite (scheduled or manual, on main), if any
# is still downloadable. Pull-request runs are not nightlies and are never compared to.
if [ -n "${GH_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
  ids=$(gh run list -R "$GITHUB_REPOSITORY" --workflow nightly-parity.yml --branch main \
    --limit 10 --json databaseId,status,event \
    --jq '.[] | select(.status == "completed" and (.event == "schedule" or .event == "workflow_dispatch")) | .databaseId' \
    2>/dev/null || true)
  for id in $ids; do
    [ "$id" = "${GITHUB_RUN_ID:-}" ] && continue
    prev=$(mktemp -d)
    if gh run download "$id" -R "$GITHUB_REPOSITORY" -n "nightly-$suite" -D "$prev" >/dev/null 2>&1 &&
      [ -f "$prev/verdicts.json" ]; then
      # Re-read that night's own reports with tonight's verdicts.py where the
      # artifact kept them, so both sides are extracted by the same rule. Its
      # stored verdicts.json was written by the rule that night ran: the first
      # night after #901 counted the stochastic ensemble, the night before had
      # timed the load alone, and the two read as a x483 slowdown.
      prev_verdicts="$prev/verdicts.json"
      if [ -f "$prev/report.json" ]; then
        prev_args=(--report "$prev/report.json")
        for r in "$prev"/report-*.json; do
          [ -f "$r" ] && prev_args+=(--report "$r")
        done
        prev_prov=$(python3 -c 'import json, sys
s = json.load(open(sys.argv[1])).get("source", {})
print(json.dumps({k: s[k] for k in ("run_id", "sha", "event") if k in s}))' "$prev/verdicts.json" 2>/dev/null || echo '{}')
        if python3 "$here/verdicts.py" extract --kind "$kind" --suite "$suite" \
          "${prev_args[@]}" --out "$prev/verdicts-now.json" --provenance "$prev_prov" >/dev/null 2>&1; then
          prev_verdicts="$prev/verdicts-now.json"
        fi
      fi
      python3 "$here/verdicts.py" diff \
        --baseline "$prev_verdicts" --fresh "$out/verdicts.json" \
        --label "vs previous night (run $id)" --flaky "$here/flaky.json" --timeout-flips-info \
        --json-out "$out/diff-prev.json" --md-out "$out/diff-prev.md" "$@" >/dev/null || rc=1
      [ -f "$out/diff-prev.md" ] && cat "$out/diff-prev.md" >>"$summary"
      break
    fi
  done
fi

exit $rc
