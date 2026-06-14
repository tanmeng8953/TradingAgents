#!/bin/zsh
set -euo pipefail

LABEL="com.tanmeng.tradingagents-web"
SCRIPT_DIR="${0:A:h}"
REPO_DIR="${SCRIPT_DIR:h}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/TradingAgents"
EXECUTABLE="$REPO_DIR/.venv/bin/tradingagents"
MIN_FREE_KIB=$((5 * 1024 * 1024))
ACTION="${1:-install}"

available_kib() {
  if [[ -n "${TRADINGAGENTS_TEST_AVAILABLE_KIB:-}" ]]; then
    print -r -- "$TRADINGAGENTS_TEST_AVAILABLE_KIB"
    return
  fi
  df -Pk "$REPO_DIR" | awk 'NR == 2 { print $4 }'
}

check_disk_space() {
  local free_kib
  free_kib="$(available_kib)"
  if [[ -z "$free_kib" || "$free_kib" -lt "$MIN_FREE_KIB" ]]; then
    print -u2 -- "TradingAgents installation requires at least 5 GiB free disk space."
    print -u2 -- "Available: $(( ${free_kib:-0} / 1024 / 1024 )) GiB. No files were deleted."
    return 2
  fi
}

render_plist() {
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$EXECUTABLE</string>
    <string>web</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8502</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$REPO_DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>TRADINGAGENTS_WORKBENCH_ENV_FILE</key>
    <string>/Users/tanmeng/GitHub/AI/ai-workbench/.env</string>
    <key>TRADINGAGENTS_RUNS_DIR</key>
    <string>$HOME/.tradingagents/runs</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/web.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/web.stderr.log</string>
</dict>
</plist>
EOF
}

install_agent() {
  check_disk_space
  if [[ ! -x "$EXECUTABLE" ]]; then
    print -u2 -- "Missing $EXECUTABLE. Run: uv sync --python 3.11 --extra dev --extra web"
    return 3
  fi

  mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
  render_plist > "$PLIST"
  plutil -lint "$PLIST" >/dev/null

  if [[ "${TRADINGAGENTS_SKIP_LAUNCHCTL:-0}" != "1" ]]; then
    launchctl bootout "gui/$UID/$LABEL" >/dev/null 2>&1 || true
    launchctl bootstrap "gui/$UID" "$PLIST"
    launchctl kickstart -k "gui/$UID/$LABEL"
  fi
  print -r -- "Installed $PLIST"
}

uninstall_agent() {
  if [[ "${TRADINGAGENTS_SKIP_LAUNCHCTL:-0}" != "1" ]]; then
    launchctl bootout "gui/$UID/$LABEL" >/dev/null 2>&1 || true
  fi
  rm -f "$PLIST"
  print -r -- "Removed $PLIST"
}

status_agent() {
  launchctl print "gui/$UID/$LABEL"
}

case "$ACTION" in
  install)
    install_agent
    ;;
  uninstall)
    uninstall_agent
    ;;
  status)
    status_agent
    ;;
  render)
    render_plist
    ;;
  *)
    print -u2 -- "Usage: $0 {install|uninstall|status|render}"
    exit 64
    ;;
esac
