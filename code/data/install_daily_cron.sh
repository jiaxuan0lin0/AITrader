#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=load_a_share_env.sh
source "$SCRIPT_DIR/load_a_share_env.sh"

JOB_SCRIPT="$ROOT_DIR/data/daily_update_a_share.sh"
ENV_FILE="$A_SHARE_ENV_FILE"
SCHEDULE="${1:-0 8 * * *}"
TMP_FILE="$(mktemp)"

cleanup() {
  rm -f "$TMP_FILE"
}
trap cleanup EXIT

EXISTING_CRON="$(crontab -l 2>/dev/null || true)"
FILTERED_CRON="$(printf '%s\n' "$EXISTING_CRON" | grep -v "$JOB_SCRIPT" || true)"
COMMAND="A_SHARE_ENV_FILE='$ENV_FILE' /bin/bash '$JOB_SCRIPT'"

printf '%s\n' "$FILTERED_CRON" | sed '/^$/N;/^\n$/D' > "$TMP_FILE"
printf '%s %s\n' "$SCHEDULE" "$COMMAND" >> "$TMP_FILE"
crontab "$TMP_FILE"

echo "Installed cron job:"
echo "$SCHEDULE $COMMAND"
