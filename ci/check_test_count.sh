#!/usr/bin/env bash
#
# Hold one CI test count to its bound in ci/test_floors.env (issue #584).
#
#   bash ci/check_test_count.sh NAME COUNT 'what crossing the bound means'
#
# NAME picks one bound from ci/test_floors.env. That is NAME_FLOOR (the count
# must not fall below it) or NAME_CEILING (it must not rise above it), exactly
# one of the two, plus NAME_MAX_SLACK: how far the count may sit from the bound
# before the bound is stale.
#
# Either direction fails the job. Crossing the bound is the false green every
# caller already guarded against (#28/#36): tests vanished, or turned into
# skips, and the run stayed green. A stale bound is what #584 added. A floor far
# below its count guards nothing. The suite's collected floor sat at 3480 while
# 11288 tests were collected, and nothing said so, because moving a bound was a
# chore nobody was made to do. Now the run that outgrows a bound is the run
# that fails, and the message names the line to edit.
#
# Exit status: 0 within bounds; 1 the bound was crossed or has gone stale; 2 a
# usage or file error (an unknown NAME, a key missing or set twice, something
# that is not an integer). Messages are GitHub ::error:: annotations on stdout,
# where the callers' own messages already go.
#
# Bash 3.2 compatible (the macOS system bash), and run under Git Bash on the
# Windows legs. The file is parsed, not sourced, so a duplicated key is an error
# rather than last-one-wins, and a CRLF checkout of it still reads.

set -euo pipefail

FLOORS="$(dirname "$0")/test_floors.env"

die() {
    echo "::error::ci/check_test_count.sh: $*"
    exit 2
}

[ "$#" -eq 3 ] || die "usage: ci/check_test_count.sh NAME COUNT 'what crossing the bound means' (got $# arguments)"
name="$1"
count="$2"
breach="$3"

[[ "$name" =~ ^[A-Z][A-Z0-9_]*$ ]] || die "NAME '$name' is not an UPPER_CASE identifier."
[[ "$count" =~ ^[0-9]+$ ]] || die "$name: count '$count' is not a non-negative integer, so nothing was counted."
[ -f "$FLOORS" ] || die "$FLOORS does not exist."
count=$((10#$count))

# lookup KEY: set REPLY to KEY's value in the file, or to "" when it is absent.
# A global rather than $(...), so that die's message reaches the log instead of
# being captured.
lookup() {
    local key="$1" n line
    REPLY=""
    n="$(grep -cE "^${key}=" "$FLOORS" || true)"
    [ "$n" -le 1 ] || die "$key is set $n times in ci/test_floors.env; keep one."
    [ "$n" -eq 1 ] || return 0
    line="$(grep -E "^${key}=" "$FLOORS")"
    line="${line%$'\r'}"
    REPLY="${line#*=}"
    [[ "$REPLY" =~ ^[0-9]+$ ]] || die "$key='$REPLY' in ci/test_floors.env is not a non-negative integer."
    REPLY=$((10#$REPLY))
}

lookup "${name}_FLOOR"
floor="$REPLY"
lookup "${name}_CEILING"
ceiling="$REPLY"
lookup "${name}_MAX_SLACK"
slack="$REPLY"

if [ -n "$floor" ] && [ -n "$ceiling" ]; then
    die "$name has both a _FLOOR and a _CEILING in ci/test_floors.env; a count is bounded in one direction."
fi
if [ -z "$floor" ] && [ -z "$ceiling" ]; then
    die "ci/test_floors.env has no ${name}_FLOOR or ${name}_CEILING."
fi
[ -n "$slack" ] || die "ci/test_floors.env has no ${name}_MAX_SLACK."

if [ -n "$floor" ]; then
    echo "$name: $count (floor $floor, at most $slack above it)."
    if [ "$count" -lt "$floor" ]; then
        echo "::error::$name is $count, below its floor of $floor: $breach."
        exit 1
    fi
    if [ $((count - floor)) -gt "$slack" ]; then
        echo "::error::${name}_FLOOR is stale: $name is $count, $((count - floor)) above the floor of $floor, and at most $slack is allowed. That many could disappear without this guard noticing. Raise ${name}_FLOOR in ci/test_floors.env; the note above it says where to set it."
        exit 1
    fi
else
    echo "$name: $count (ceiling $ceiling, at most $slack below it)."
    if [ "$count" -gt "$ceiling" ]; then
        echo "::error::$name is $count, above its ceiling of $ceiling: $breach."
        exit 1
    fi
    if [ $((ceiling - count)) -gt "$slack" ]; then
        echo "::error::${name}_CEILING is stale: $name is $count, $((ceiling - count)) below the ceiling of $ceiling, and at most $slack is allowed. That many more could appear without this guard noticing. Lower ${name}_CEILING in ci/test_floors.env; the note above it says where to set it."
        exit 1
    fi
fi
