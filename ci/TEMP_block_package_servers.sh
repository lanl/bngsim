#!/usr/bin/env bash
# TEMPORARY, reverted before merge (#1488). Makes ccache's install fail the way
# 2026-11-11's outage did (Ubuntu's package servers did not resolve), to show
# that every build still completes without ccache. On macOS it blocks the hosts
# Homebrew installs from.
set -euo pipefail
if [ "$RUNNER_OS" = Linux ]; then
  hosts="archive.ubuntu.com azure.archive.ubuntu.com security.ubuntu.com"
  probe=http://archive.ubuntu.com/ubuntu/
else
  hosts="formulae.brew.sh ghcr.io pkg-containers.githubusercontent.com"
  probe=https://ghcr.io/v2/
fi
if command -v ccache; then echo "::warning::ccache is already on PATH here, so this leg cannot show the fallback"; fi
# Control: the probe answers before the block...
curl -sS -m 15 -o /dev/null "$probe" || echo "::warning::$probe did not answer even before the block"
for h in $hosts; do echo "0.0.0.0 $h" | sudo tee -a /etc/hosts >/dev/null; done
if [ "$RUNNER_OS" = macOS ]; then sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder || true; fi
# ...and must not after it.
if curl -sS -m 15 -o /dev/null "$probe"; then echo "::error::the block did not take: $probe answered"; exit 1; fi
echo "blocked: $hosts"
