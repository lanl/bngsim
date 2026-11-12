#!/usr/bin/env python3
"""Which jobs of a workflow run did GitHub never start? (#1488)

When GitHub has no hosted runner to give a job ("The job was not acquired by
Runner of type hosted even after multiple attempts", in November 2026 a shortage
of macOS arm64 runners), the job ends cancelled without having run a step, and
GitHub marks the whole run failed. Nothing in the job failed, and nothing can be
said about the code: the job is simply missing. forced-jit.yml's Report job and
forced-jit-rerun.yml ask this script the same question, so they cannot disagree
about the answer: did the run fail only because jobs never started?

    gh api "repos/OWNER/REPO/actions/runs/RUN/attempts/N/jobs?per_page=100" \\
      | python3 ci/unstarted_jobs.py --prefix "Python suite"

It reads GitHub's job listing on stdin. Among the jobs whose name starts with
``--prefix``, it prints each one that never started, its id and name on a line,
when those are the only ones that did not succeed. It prints nothing when they
all succeeded, and nothing when any of them ran and failed or was cancelled once
running: a run with a result of its own to report.

A job that never started is one with the conclusion "cancelled", no runner and
no steps. A job a person cancels while it waits for a runner looks the same, but
the run then ends "cancelled", not "failure", and forced-jit-rerun.yml re-runs
only failed runs.

Exit status: 0 when the listing was read, whatever it holds; 2 when it could not
be (not JSON, not a job listing, or only one page of a longer one).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable

#: Conclusions that need no re-run. GitHub reports a job skipped by its ``if:`` as
#: "skipped".
_FINE = frozenset({"success", "skipped"})


def never_started(job: dict) -> bool:
    """GitHub gave the job no runner, so it ran nothing."""
    return (
        job.get("conclusion") == "cancelled" and not job.get("runner_id") and not job.get("steps")
    )


def unstarted(jobs: Iterable[dict], prefix: str) -> list[dict]:
    """The jobs named ``prefix…`` that never started, when those are the only
    ones of them that did not succeed; otherwise none."""
    mine = [j for j in jobs if str(j.get("name", "")).startswith(prefix)]
    missing = [j for j in mine if j.get("conclusion") not in _FINE]
    if missing and all(never_started(j) for j in missing):
        return missing
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prefix", required=True, help="consider only jobs whose name starts with this"
    )
    args = parser.parse_args(argv)
    try:
        listing = json.load(sys.stdin)
        jobs = listing["jobs"]
        total = int(listing.get("total_count", len(jobs)))
        if not isinstance(jobs, list) or not all(isinstance(j, dict) for j in jobs):
            raise TypeError("'jobs' is not a list of jobs")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"unstarted_jobs: not a job listing: {exc}", file=sys.stderr)
        return 2
    if total > len(jobs):
        # A job on a later page could be one that failed.
        print(f"unstarted_jobs: the listing holds {len(jobs)} of {total} jobs", file=sys.stderr)
        return 2
    for job in unstarted(jobs, args.prefix):
        print(f"{job.get('id')}\t{job.get('name')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
