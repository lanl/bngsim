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
#      regresses before anyone re-baselined it. A row the committed baseline
#      re-recorded since that night's commit, which tonight matches, is reported there
#      and not alerted: it was reviewed when it was recorded.
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
  runs=$(gh run list -R "$GITHUB_REPOSITORY" --workflow nightly-parity.yml --branch main \
    --limit 10 --json databaseId,status,event,headSha \
    --jq '.[] | select(.status == "completed" and (.event == "schedule" or .event == "workflow_dispatch")) | "\(.databaseId) \(.headSha)"' \
    2>/dev/null || true)
  while read -r id sha <&3; do
    [ -n "$id" ] || continue
    [ "$id" = "${GITHUB_RUN_ID:-}" ] && continue
    prev=$(mktemp -d)
    if gh run download "$id" -R "$GITHUB_REPOSITORY" -n "nightly-$suite" -D "$prev" >/dev/null 2>&1 &&
      [ -f "$prev/verdicts.json" ]; then
      # The committed baseline as of that night's commit (a shallow checkout fetches
      # it); without it every change between the two nights is judged as before.
      then_args=()
      if git -C "$here" fetch -q --depth=1 origin "$sha" 2>/dev/null &&
        git -C "$here" show "$sha:./baselines/$suite.json" >"$prev/baseline-then.json" 2>/dev/null; then
        then_args=(--baseline-then "$prev/baseline-then.json" --baseline-now "$here/baselines/$suite.json")
      fi
      python3 "$here/verdicts.py" diff \
        --baseline "$prev/verdicts.json" --fresh "$out/verdicts.json" \
        --label "vs previous night (run $id)" --flaky "$here/flaky.json" \
        --timeout-flips-info --phases-may-differ ${then_args[@]+"${then_args[@]}"} \
        --json-out "$out/diff-prev.json" --md-out "$out/diff-prev.md" "$@" >/dev/null || rc=1
      [ -f "$out/diff-prev.md" ] && cat "$out/diff-prev.md" >>"$summary"
      break
    fi
  done 3<<<"$runs"
fi

exit $rc
