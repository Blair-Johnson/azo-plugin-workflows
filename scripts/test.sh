#!/usr/bin/env bash
set -euo pipefail
# Tests run inside the Agent Zoo host environment, which supplies agent_zoo and PyYAML.
: "${AZO_HOST_MANIFEST:=$HOME/Documents/GitHub/agent-zoo/pixi.toml}"
exec env -u PYTHONPATH -u PYTHONHOME pixi run --manifest-path "$AZO_HOST_MANIFEST" --frozen --no-install python -m pytest -q "$@"
