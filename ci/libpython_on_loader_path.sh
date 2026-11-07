#!/usr/bin/env bash
# Let libroadrunner find the interpreter's libpython on a Linux runner.
#
#   ci/libpython_on_loader_path.sh [PYTHON]    (default: .venv/bin/python)
#
# libroadrunner's Linux wheel links _roadrunner.so against libpython3.12.so.1.0
# by name and carries no path to it. On ubuntu-24.04 the system Python is 3.12,
# so that file is in /usr/lib and the import works by accident. On ubuntu-26.04
# the system Python is 3.14, uv downloads its own 3.12, and `import roadrunner`
# fails with "libpython3.12.so.1.0: cannot open shared object file". Every test
# that needs RoadRunner then skips (conftest declares that skip as expected for
# a developer-only engine), and the parity jobs that run it fail.
#
# uv's Python ships the library in its own LIBDIR. When the system loader cannot
# already resolve the interpreter's shared-library name, this appends that LIBDIR
# to LD_LIBRARY_PATH for the rest of the job, through $GITHUB_ENV. Where the
# system has the library (ubuntu-24.04, or any host whose Python is the system's)
# it changes nothing. Not Linux: nothing to do.
set -euo pipefail

[ "$(uname -s)" = Linux ] || exit 0

python=${1:-.venv/bin/python}
read -r libdir soname < <(
  "$python" -c 'import sysconfig as s; print(s.get_config_var("LIBDIR") or "-", s.get_config_var("INSTSONAME") or "-")'
)

if [ "$soname" = - ]; then
  echo "libpython: $python reports no shared-library name; nothing to do"
  exit 0
fi
ldconfig=$(command -v ldconfig || echo /sbin/ldconfig)
if "$ldconfig" -p | awk -v s="$soname" '$1 == s { found = 1 } END { exit !found }'; then
  echo "libpython: the system loader already finds $soname; nothing to do"
  exit 0
fi
if [ ! -f "$libdir/$soname" ]; then
  echo "::warning::libpython: $soname is neither on the system loader path nor in $libdir; RoadRunner will not import"
  exit 0
fi
if [ -z "${GITHUB_ENV:-}" ]; then
  echo "libpython: $soname is in $libdir only; outside GitHub Actions, run with"
  echo "  export LD_LIBRARY_PATH=$libdir\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
  exit 0
fi
echo "LD_LIBRARY_PATH=$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" >>"$GITHUB_ENV"
echo "libpython: $soname is in $libdir only; added it to LD_LIBRARY_PATH for this job"
